#!/usr/bin/env python3
# =====================================================================
#  paper_trade_ibkr.py  –  hardened day & night wrapper for IBKR
#  ▸ Per-model thresholds: 1-layer = 0.70 · 2-layer = 0.80
#  ▸ PDT guard, buy-ledger, single-instance lock
#  ▸ **IBKR only for order-routing — market data comes from yfinance**
# =====================================================================
import os, sys, time, warnings, pickle, re, contextlib, json, datetime as dt
from pathlib import Path
from typing  import Dict

import numpy as np
import pandas as pd
import yfinance as yf
import torch, torch.nn as nn
from ib_insync import IB, Stock, MarketOrder                 # ≠ market-data calls!

# ────────── USER CONFIG ──────────
SEQ_LEN, HIDDEN_SIZE = 5, 512
TICKERS: Dict[str, str] = {
    "FBK":"XLF", #much more ticker-sector pairs are here in the actual implementation but I removed them here for privacy
}
NIGHT_ONLY = {"MPC","TSLA"}
MODELS_DIR = Path.home()/ "Documents"/ "QuantProject"

# ────────── IB connection parameters ──────────
IB_HOST = os.getenv("IB_HOST","127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT",7497))          # paper-TWS default
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID",1))

# ────────── CLI arg ──────────
if len(sys.argv)!=2 or sys.argv[1] not in {"open","close"}:
    sys.exit("Usage: paper_trade_ibkr.py  open|close")
SESSION = sys.argv[1]

# ────────── single-instance lock ──────────
LOCK = Path(f"/tmp/paper_trade_ibkr_{SESSION}.lock")
if LOCK.exists():
    print("❕ another instance is already running — aborting."); sys.exit(0)
LOCK.touch()

# ────────── PDT buy-ledger helpers ──────────
LEDGER_DIR = Path("/tmp")
today_tag  = dt.date.today().isoformat()
ledger_path = LEDGER_DIR / f"ibkr_buys_{today_tag}.json"
BUY_LEDGER : Dict[str,int] = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}

def save_ledger(): ledger_path.write_text(json.dumps(BUY_LEDGER))

# ────────── IB connection (orders only) ──────────
ib = IB()
try:
    ib.connect(IB_HOST,IB_PORT,IB_CLIENT_ID,timeout=5)
except Exception as e:
    LOCK.unlink(missing_ok=True); sys.exit(f"❌  IBKR connect failed: {e}")

# PDT snapshot ----------------------------------------------------------
summ = {r.tag:r.value for r in ib.accountSummary()}
equity   = float(summ.get("NetLiquidation",0))
day_left = float(summ.get("DayTradesRemainingT+0",summ.get("DayTradesRemaining",-1)))
PDT_LOCK = (equity < 25_000) and (day_left is not None and day_left <= 0)
if PDT_LOCK:
    print(f"🚫 PDT lock (equity={equity:.2f}  remaining_day_trades={day_left})")

# ────────── LSTM helper ──────────
class LSTMNet(nn.Module):
    def __init__(self,n_in,h,layers):
        super().__init__()
        self.lstm = nn.LSTM(n_in,h,layers,batch_first=True,dropout=0.2 if layers>1 else 0.)
        self.fc   = nn.Linear(h,1)
    def forward(self,x):
        _,(h,_) = self.lstm(x)
        return torch.sigmoid(self.fc(h[-1])).squeeze()

def n_layers(state):                                     # detect 1- vs 2-layer
    idx={int(m.group(1)) for k in state
         for m in [re.search(r"weight_ih_l(\d+)",k)] if m}
    return max(idx)+1 if idx else 1
def thr(lay): return 0.7 if lay==1 else 0.8

# ────────── model/scaler/threshold maps ──────────
DAY_M, NIGHT_M = {}, {}
DAY_S, NIGHT_S = {}, {}
DAY_THR, NIGHT_THR = {}, {}

for s,sec in TICKERS.items():
    mp,sp = MODELS_DIR/f"best_model_{s}.pth", MODELS_DIR/f"scaler_{s}.pkl"
    if mp.exists() and sp.exists():
        mu,sig = pickle.load(open(sp,'rb'))
        st     = torch.load(mp,map_location='cpu')
        lay    = n_layers(st)
        net    = LSTMNet(len(mu),HIDDEN_SIZE,lay); net.load_state_dict(st); net.eval()
        DAY_M[s], DAY_S[s], DAY_THR[s] = net,(mu,sig),thr(lay)
        print(f"ℹ️  {s}: {lay}-layer DAY  (thr={DAY_THR[s]:.2f})")
    if s in NIGHT_ONLY:
        mpn,spn = MODELS_DIR/f"NIGHT_best_model_{s}.pth", MODELS_DIR/f"NIGHT_scaler_{s}.pkl"
        if mpn.exists() and spn.exists():
            mu,sig = pickle.load(open(spn,'rb'))
            st = torch.load(mpn,map_location='cpu')
            lay=n_layers(st)
            net=LSTMNet(len(mu),HIDDEN_SIZE,lay); net.load_state_dict(st); net.eval()
            NIGHT_M[s], NIGHT_S[s], NIGHT_THR[s] = net,(mu,sig),thr(lay)
            print(f"ℹ️  {s}: {lay}-layer NIGHT (thr={NIGHT_THR[s]:.2f})")

# ────────── yfinance helpers ──────────
def yf_feat(sym,sec):
    try:
        df=yf.download([sym,sec],period=f"{SEQ_LEN+1}d",interval="1d",
                       group_by='ticker',progress=False)
    except Exception as e:
        warnings.warn(f"yfinance {sym}: {e}"); return None
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=['_'.join(c) for c in df.columns]
    df=df.dropna()
    if len(df)<SEQ_LEN: return None
    cols=[f"{sym}_{c}" for c in ("Open","High","Low","Close","Volume")]
    cols+=[f"{sec}_{c}" for c in ("Open","High","Low","Close","Volume")
           if f"{sec}_{c}" in df.columns]
    return df[cols].values[-SEQ_LEN:]

def predict(sym,night=False):
    if night and sym in NIGHT_M:
        mu,sig = NIGHT_S[sym]; net=NIGHT_M[sym]
    else:
        mu,sig = DAY_S[sym];   net=DAY_M[sym]
    X = yf_feat(sym,TICKERS[sym])
    if X is None: return 0.0
    X=(X-mu)/sig
    with torch.no_grad():
        return float(net(torch.tensor(X[None],dtype=torch.float32)))

# ONLY yfinance for prices ---------------------------------------------
def price(sym):
    """Return best-effort last price using yfinance only."""
    try:
        info = yf.Ticker(sym).info
        px   = info.get("regularMarketPrice") or info.get("previousClose")
        if px and not np.isnan(px): return float(px)
    except Exception:
        pass
    warnings.warn(f"price(): no quote for {sym}; defaulting to 1 USD")
    return 1.0

# IB wrappers (orders only) --------------------------------------------
def stk(sym): return Stock(sym,'SMART','USD')

def positions() -> Dict[str,int]:
    return {p.contract.symbol:int(p.position) for p in ib.positions() if p.position!=0}

def cancel_open(sym):
    for t in ib.openTrades():
        if t.contract.symbol==sym: ib.cancelOrder(t.order)

def safe_sell(sym,q,tries=3):
    if q<=0: return
    contract=stk(sym); cancel_open(sym)
    for i in range(tries):
        try:
            print(f"✓ SELL {q} {sym}")
            ib.placeOrder(contract,MarketOrder('SELL',q)); ib.sleep(0.4); return
        except Exception as e:
            print(f"… retry {i+1}/{tries} {sym}: {e}"); ib.sleep(1)
    print(f"⚠️  failed to sell {sym}")

def safe_buy(sym,q):
    if q<=0: return
    try:
        print(f"✓ BUY {q} {sym}")
        ib.placeOrder(stk(sym),MarketOrder('BUY',q))
        if SESSION=='open':
            BUY_LEDGER[sym]=BUY_LEDGER.get(sym,0)+q; save_ledger()
    except Exception as e:
        print(f"⚠️  buy failed {sym}: {e}")

def avail_cash():
    sm = {r.tag:r.value for r in ib.accountSummary()}
    return float(sm.get("AvailableFunds",sm.get("AvailableFunds-C",0)))

# rebalancer -----------------------------------------------------------
def rebalance(wts:Dict[str,float]):
    cur=positions()
    if PDT_LOCK and SESSION=='open':
        print("PDT lock: skipping sells."); to_sell={}
    else:
        to_sell={s:q for s,q in cur.items() if s not in wts}
    for s,q in to_sell.items(): safe_sell(s,q)
    ib.sleep(1)
    cash=avail_cash()
    for sym,w in wts.items():
        q=int((cash*w)//price(sym))
        if q>0 and (not (PDT_LOCK and SESSION=='open')):
            safe_buy(sym,q)

# main -----------------------------------------------------------------
def run():
    sel={}
    if SESSION=='open':
        for s in TICKERS:
            if predict(s)>=DAY_THR.get(s,0.8): sel[s]=True
    else:
        for s in NIGHT_ONLY:
            if predict(s,night=True)>=NIGHT_THR.get(s,0.8): sel[s]=True
    if not sel: wts={'SPY':1.0}
    else:
        raw={s:predict(s,SESSION=='close' and s in NIGHT_ONLY) for s in sel}
        tot=sum(raw.values()); wts={k:v/tot for k,v in raw.items()}
    print("Target weights:",wts)
    rebalance(wts)

    if SESSION=='close':
        extra={s:q for s,q in positions().items() if s not in NIGHT_ONLY}
        if PDT_LOCK:
            extra={s:q for s,q in extra.items() if s not in BUY_LEDGER}
        for s,q in extra.items(): safe_sell(s,q)
        ib.sleep(1)
        spq=int(avail_cash()//price('SPY'))
        if spq>0: safe_buy('SPY',spq)

run()

# cleanup --------------------------------------------------------------
with contextlib.suppress(Exception): LOCK.unlink()
ib.disconnect()
