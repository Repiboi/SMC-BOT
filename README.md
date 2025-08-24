This is a MetaTrader 5 (MT5) automated trading bot built for multi-symbol scanning and smart execution. Its purpose is to continuously analyze different forex pairs, commodities, indices, and even crypto pairs (like BTCUSDm) and execute trades based on high-confidence strategies.

🔑 Key Features:

1. Multi-Symbol Scanning

a. The bot cycles through all available market symbols (e.g., XAUUSDm, EURUSDm, BTCUSDm, US30m, etc.)

b. Filters out unwanted pairs if needed (e.g., skipping crypto, metals, etc. if specified).

c. Works with hundreds of symbols without slowing down.


2. Smart Strategies (SMC-Inspired)

a.Break of structure (BoS) detection.

b. Trendline breakouts.

c. Order block & supply/demand identification.

d. Liquidity grabs with wick rejection.

e. Volume surge confirmation.

f. The bot only takes trades when multiple strategies agree for higher accuracy.


3. Risk Management

a. Dynamic lot sizing based on ATR (volatility).

b. Per-trade stop-loss and take-profit levels.

c. Trailing stop logic to lock profits as the trade moves in your favor.

d. Equity protection: The bot automatically shuts down if the drawdown exceeds 10% of account equity.


4. Continuous Profit Mode

a. Unlike older versions that stopped after hitting profit, this bot keeps running until:

b. You manually stop it, or

c. The account hits the drawdown safety net.


5. MT5 Integration

a. Asks for login, password, and server when starting.

b. Displays account balance, equity, and trading actions in real-time.

c. Handles order modifications (SL/TP updates) and new trades automatically.
