1.3 is the template for generating scalar files used to generate a threshold likelihood of stock price movement up, and we place trades when that threshold is above a certain %. It also computes a p-value to see the likelihood that the predictions were pure chance or actually created an edge. 
1.4 is a monte carlo simulation of the strategy in recent years.
paper_trade_ibkr.py is the trade logic to implement the strategy (within Trader Workstation).
The actual implementation of the strategy requires more files (which are not public)
