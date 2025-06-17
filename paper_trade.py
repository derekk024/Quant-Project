#!/usr/bin/env python3
"""
Minimal live-trading script for GitHub Actions.

Usage:
    python paper_trade.py open   # run at 09:30 ET
    python paper_trade.py close  # run at 15:59 ET
"""

import os, sys, pickle, datetime, pytz
import yfinance as yf
import numpy as np
import torch
import alpaca_trade_api as tradeapi

# ─── Config ────────────────────────────────────────────────────────────
TICKER = "FBK"                         # keep in sync with model filename
MODEL_FILE  = f"best_model_{TICKER}.pth"
SCALER_FILE = f"scaler_{TICKER}.pkl"
THRESHOLD   = 0.80                     # buy only if P(up) ≥ 0.80
SEQ_LEN     = 5

# ─── Helper: load model if it exists ───────────────────────────────────
class LSTMClassifier(torch.nn.Module):
    def __init__(self, in_sz, h=512, n_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(in_sz, h, n_layers,
                                  batch_first=True, dropout=0.2)
        self.fc   = torch.nn.Linear(h, 1)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h[-1])).squeeze()

def load_model_and_scaler():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
        print("⚠️  Model/scaler missing; running in NO-TRADE mode.")
        return None, None, None
    mu, sig = pickle.load(open(SCALER_FILE, "rb"))
    model = LSTMClassifier(len(mu))
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    model.eval()
    return model, mu, sig

# ─── Alpaca connection ────────────────────────────────────────────────
API_KEY = os.getenv("ALPACA_API_KEY_ID")
API_SEC = os.getenv("ALPACA_SECRET_KEY")
api = tradeapi.REST(API_KEY, API_SEC, "https://paper-api.alpaca.markets", api_version="v2")

# ─── Convenience: current Eastern time stamp for prints ───────────────
eastern = pytz.timezone("US/Eastern")
now_et  = datetime.datetime.now(eastern).strftime("%Y-%m-%d %H:%M:%S")

# ─── Main logic ────────────────────────────────────────────────────────
if len(sys.argv) != 2 or sys.argv[1] not in ("open", "close"):
    sys.exit("Usage: python paper_trade.py [open|close]")

action = sys.argv[1]
print(f"[{now_et}] Running {action.upper()} logic…", flush=True)

if action == "open":
    # 1) fetch last 5 days of OHLCV for features
    bars = yf.download(TICKER, period="6d", interval="1d", progress=False)
    if len(bars) < SEQ_LEN:
        sys.exit("Not enough bars; skip.")
    X_raw = bars[["Open","High","Low","Close","Volume"]].values[-SEQ_LEN:]
    model, mu, sig = load_model_and_scaler()
    if model is None:
        sys.exit(0)        # no model ⇒ skip trade

    X_norm = (X_raw - mu) / sig
    with torch.no_grad():
        p_up = model(torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0)).item()
    print(f"P(up) = {p_up:.2%}")

    if p_up >= THRESHOLD:
        # check we’re flat
        try:
            pos = api.get_position(TICKER)
            print("Already long; skip buy.") ; sys.exit(0)
        except tradeapi.rest.APIError:
            pass  # no position → continue
        api.submit_order(symbol=TICKER, qty=1,
                         side="buy", type="market", time_in_force="day")
        print("BUY 1 market order submitted")
    else:
        print("Signal below threshold; no trade.")

elif action == "close":
    # liquidate if we have a long
    try:
        pos = api.get_position(TICKER)
        qty = int(pos.qty)
    except tradeapi.rest.APIError:
        print("No position to close.")
        sys.exit(0)

    if qty > 0:
        api.submit_order(symbol=TICKER, qty=qty,
                         side="sell", type="market", time_in_force="day")
        print(f"SELL {qty} market order submitted")
    else:
        print("Position qty = 0; nothing to do.")
