import MetaTrader5 as mt5
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from math import isfinite

# ====== CONFIG ======
TIMEFRAME = mt5.TIMEFRAME_M1
HTF_TIMEFRAME = mt5.TIMEFRAME_H1
ATR_TF = mt5.TIMEFRAME_M5

WATCHLIST = [
    # include your broker's suffix (e.g., "m") if needed
    "XAUUSDm", "EURUSDm", "BTCUSDm", "GBPUSDm", "USDJPYm", "US30m",
]

RISK_PER_TRADE_PCT = 0.01        # 1% risk per trade
DAILY_EQUITY_TARGET_PCT = 0.03   # stop when up +3% on the day (adjust)
MAX_DRAWDOWN_PCT = 0.10          # kill-switch at -10% from session start
MAX_OPEN_POSITIONS_PER_SYMBOL = 999
MAX_TOTAL_POSITIONS = 20
COOLDOWN_SECONDS = 120            # per-symbol cooldown after fill
POLL_SECONDS = 2.5                # main loop sleep
LOG_EVERY = 30                    # print status every N loops

# ATR-based SL/TP (favorable R:R)
ATR_PERIOD = 14
SL_ATR_MULT = 1.0
TP_ATR_MULT = 2.0                 # 1:2 R:R

# ATR trailing stop update cadence
TRAIL_ATR_MULT = 1.2
TRAIL_UPDATE_SEC = 15

# SMC / ICT heuristic params
SWING_LOOKBACK = 50
ORDERBLOCK_LOOKBACK = 40
LIQUIDITY_WICK_MULT = 1.5
ZONE_BUFFER = 0.0015

MAGIC = 987654
DEVIATION = 30

# ====== UTILITIES ======
def now_ts():
    return pd.Timestamp.utcnow().tz_localize(None)

def round_to_step(value, step):
    if step <= 0 or not isfinite(step):
        return value
    return round((value / step)) * step

def safe_symbol_info(symbol):
    info = mt5.symbol_info(symbol)
    if not info:
        raise RuntimeError(f"[SymbolInfo] Could not get symbol info for {symbol}.")
    return info

def get_symbol_data(symbol, timeframe, bars=300):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"[ERROR] No price data for {symbol}. MT5: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

# ====== SMC HEURISTICS ======
def find_recent_swings(df, lookback=SWING_LOOKBACK):
    end = len(df)
    sub = df.iloc[max(0, end - lookback):end].reset_index(drop=True)
    idx_low = int(sub["low"].idxmin())
    idx_high = int(sub["high"].idxmax())
    low_val = float(sub.loc[idx_low, "low"])
    high_val = float(sub.loc[idx_high, "high"])
    orig_idx_low = max(0, end - lookback) + idx_low
    orig_idx_high = max(0, end - lookback) + idx_high
    return (orig_idx_low, low_val), (orig_idx_high, high_val)

def is_break_of_structure(df):
    if df is None or len(df) < SWING_LOOKBACK + 5:
        return None
    (swing_low_idx, swing_low), (swing_high_idx, swing_high) = find_recent_swings(df, SWING_LOOKBACK)
    prev_close = float(df["close"].iloc[-2])
    last_close = float(df["close"].iloc[-1])
    if prev_close <= swing_high and last_close > swing_high:
        return "buy"
    if prev_close >= swing_low and last_close < swing_low:
        return "sell"
    return None

def detect_liquidity_sweep(df):
    if df is None or len(df) < 30:
        return None
    recent = df.iloc[-30:]
    last = df.iloc[-1]
    last_body_high = max(last["open"], last["close"])
    last_body_low = min(last["open"], last["close"])
    upper_wick = last["high"] - last_body_high
    lower_wick = last_body_low - last["low"]

    avg_upper = (recent["high"] - recent[["open", "close"]].max(axis=1)).abs().mean()
    avg_lower = (recent[["open", "close"]].min(axis=1) - recent["low"]).abs().mean()

    if avg_lower > 0 and lower_wick > avg_lower * LIQUIDITY_WICK_MULT and last["close"] > last["open"]:
        return "buy"
    if avg_upper > 0 and upper_wick > avg_upper * LIQUIDITY_WICK_MULT and last["close"] < last["open"]:
        return "sell"
    return None

def detect_order_block(df):
    if df is None or len(df) < ORDERBLOCK_LOOKBACK + 5:
        return None
    window = df.iloc[-ORDERBLOCK_LOOKBACK:]
    bodies = (window["close"] - window["open"]).abs()
    avg_body = bodies.mean()
    filt = bodies > avg_body * 1.6
    if not filt.any():
        return None
    large_idx = int(bodies[filt].idxmax())
    candle = df.loc[large_idx]
    body_high = max(candle["open"], candle["close"])
    body_low = min(candle["open"], candle["close"])
    last_close = float(df["close"].iloc[-1])
    buffer = body_high * ZONE_BUFFER

    if candle["close"] < candle["open"]:  # bearish large -> bullish OB on return
        if body_low - buffer <= last_close <= body_high + buffer and df["close"].iloc[-1] > df["open"].iloc[-1]:
            return "buy"
    else:  # bullish large -> bearish OB on return
        if body_low - buffer <= last_close <= body_high + buffer and df["close"].iloc[-1] < df["open"].iloc[-1]:
            return "sell"
    return None

def detect_supply_demand_zone(df):
    if df is None or len(df) < 60:
        return None
    look = df.iloc[-60:]
    avg_high = look["high"].nlargest(3).mean()
    avg_low = look["low"].nsmallest(3).mean()
    last_close = float(df["close"].iloc[-1])
    high_buffer = avg_high * ZONE_BUFFER
    low_buffer = avg_low * ZONE_BUFFER
    if abs(last_close - avg_low) <= low_buffer and df["close"].iloc[-1] > df["open"].iloc[-1]:
        return "buy"
    if abs(last_close - avg_high) <= high_buffer and df["close"].iloc[-1] < df["open"].iloc[-1]:
        return "sell"
    return None

def smc_signal(symbol):
    df = get_symbol_data(symbol, TIMEFRAME, bars=220)
    if df is None or len(df) < 120:
        return None, {}

    reasons = {}
    bos = is_break_of_structure(df);            reasons["bos"] = bos
    sweep = detect_liquidity_sweep(df);         reasons["liquidity"] = sweep
    ob = detect_order_block(df);                reasons["orderblock"] = ob
    zone = detect_supply_demand_zone(df);       reasons["zone"] = zone

    votes = [v for v in [bos, sweep, ob, zone] if v is not None]
    if len(votes) >= 2 and len(set(votes)) == 1:
        return votes[0], reasons
    # single strong signal will be accepted only with HTF alignment (checked later)
    if bos is not None:
        return bos, reasons  # mark as provisional; HTF filter will decide
    if sweep is not None:
        return sweep, reasons
    return None, reasons

# ====== INDICATORS & RISK ======
def atr_value(symbol, timeframe=ATR_TF, period=ATR_PERIOD):
    df = get_symbol_data(symbol, timeframe, bars=period + 5)
    if df is None or len(df) < period + 1:
        return None
    h_l = df["high"] - df["low"]
    h_c = (df["high"] - df["close"].shift()).abs()
    l_c = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else None

def htf_trend(symbol):
    df = get_symbol_data(symbol, HTF_TIMEFRAME, bars=60)
    if df is None or len(df) < 21:
        return None
    ma_fast = df["close"].rolling(10).mean().iloc[-1]
    ma_slow = df["close"].rolling(20).mean().iloc[-1]
    if pd.isna(ma_fast) or pd.isna(ma_slow):
        return None
    if ma_fast > ma_slow: return "buy"
    if ma_fast < ma_slow: return "sell"
    return None

def price_now(symbol):
    t = mt5.symbol_info_tick(symbol)
    if not t: return None
    return float(t.ask), float(t.bid)

def compute_sl_tp(symbol, side, atr):
    ask, bid = price_now(symbol)
    if ask is None: return None, None
    entry = ask if side == "buy" else bid
    sl_dist = atr * SL_ATR_MULT
    tp_dist = atr * TP_ATR_MULT
    sl = entry - sl_dist if side == "buy" else entry + sl_dist
    tp = entry + tp_dist if side == "buy" else entry - tp_dist
    return float(sl), float(tp)

def risk_based_lot(symbol, balance, atr):
    info = safe_symbol_info(symbol)
    tick_val = info.trade_tick_value
    tick_size = info.trade_tick_size
    # approximate value-per-point of atr using tick_size scaling
    if tick_size <= 0 or tick_val <= 0 or atr is None:
        # fallback: minimal lot
        step = info.volume_step or 0.01
        return max(info.volume_min or 0.01, step)

    # risk amount in account currency
    risk_amt = balance * RISK_PER_TRADE_PCT
    # SL distance in price = atr * SL_ATR_MULT
    sl_price_dist = atr * SL_ATR_MULT
    # convert price distance to ticks
    ticks = max(1.0, sl_price_dist / tick_size)
    # value risk per 1 lot = ticks * tick_value * contract factor already in tick_value
    # so lot = risk_amt / (ticks * tick_val)
    raw_lot = risk_amt / (ticks * tick_val)

    # clamp to symbol constraints
    minv = info.volume_min or 0.01
    maxv = info.volume_max or 10.0
    step = info.volume_step or 0.01
    lot = min(max(raw_lot, minv), maxv)
    lot = round_to_step(lot, step)
    return float(lot)

# ====== TRADING ======
def can_open_new(symbol):
    pos = mt5.positions_get(symbol=symbol, group="")
    all_pos = mt5.positions_get()
    if not pos: return len(all_pos) < MAX_TOTAL_POSITIONS
    return len(pos) < MAX_OPEN_POSITIONS_PER_SYMBOL and len(all_pos) < MAX_TOTAL_POSITIONS

def place_order(symbol, side, lot, sl=None, tp=None):
    ask, bid = price_now(symbol)
    if ask is None:
        print(f"[{symbol}] No tick.")
        return None

    info = safe_symbol_info(symbol)
    if not info.visible:
        mt5.symbol_select(symbol, True)

    price = ask if side == "buy" else bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": float(sl) if sl else 0.0,
        "tp": float(tp) if tp else 0.0,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "SMC-Pro",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(request)
    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[{symbol}] ORDER {side.upper()} lot={lot} @ {price} -> ticket={getattr(res, 'order', None)}")
        return res
    print(f"[{symbol}] Order failed: ret={getattr(res,'retcode',None)} err={mt5.last_error()}")
    return None

def modify_sl(symbol, position_ticket, new_sl):
    pos_list = mt5.positions_get(ticket=position_ticket)
    if not pos_list:
        return
    pos = pos_list[0]
    req = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": int(pos.ticket),
        "symbol": symbol,
        "sl": float(new_sl),
        "tp": float(pos.tp),
        "magic": MAGIC,
    }
    r = mt5.order_send(req)
    if r and r.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[{symbol}] SL modified for ticket {position_ticket} -> {new_sl}")
    else:
        print(f"[{symbol}] SL modify failed: ret={getattr(r,'retcode',None)} {mt5.last_error()}")

def close_position_ticket(symbol, ticket):
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        return
    p = pos[0]
    side = mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = price_now(symbol)
    if price is None:
        return
    price = price[1] if side == mt5.ORDER_TYPE_SELL else price[0]
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": p.volume,
        "type": side,
        "position": int(ticket),
        "price": float(price),
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "SMC-Pro-close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    r = mt5.order_send(req)
    print(f"[{symbol}] Close ticket {ticket}: ret={getattr(r,'retcode',None)}")

def close_all(symbol=None):
    if symbol:
        positions = mt5.positions_get(symbol=symbol)
    else:
        positions = mt5.positions_get()
    if not positions:
        return
    for p in positions:
        close_position_ticket(p.symbol, p.ticket)

# ====== RUNTIME STATE ======
class RuntimeState:
    def __init__(self, start_equity):
        self.start_equity = start_equity
        self.last_trade_time = {s: None for s in WATCHLIST}
        self.last_trail_update = {s: None for s in WATCHLIST}
        self.loop = 0

    def cooldown_ok(self, symbol):
        t = self.last_trade_time.get(symbol)
        if not t: return True
        return (datetime.utcnow() - t).total_seconds() >= COOLDOWN_SECONDS

    def mark_trade(self, symbol):
        self.last_trade_time[symbol] = datetime.utcnow()

    def trail_due(self, symbol):
        t = self.last_trail_update.get(symbol)
        if not t: return True
        return (datetime.utcnow() - t).total_seconds() >= TRAIL_UPDATE_SEC

    def mark_trail(self, symbol):
        self.last_trail_update[symbol] = datetime.utcnow()

# ====== MAIN LOOP ======
def run():
    print("Please provide your MT5 login credentials:")
    try:
        login = int(input("Enter MT5 Login ID: ").strip())
    except ValueError:
        print("Invalid login. Must be a number. Exiting.")
        return
    password = input("Enter MT5 Password: ")
    server = input("Enter Server Name: ")

    if not mt5.initialize(login=login, password=password, server=server):
        print(f"MT5 init failed: {mt5.last_error()}")
        return
    print("MT5 initialized.")

    # ensure symbols visible
    symbols_ok = []
    all_syms = mt5.symbols_get()
    all_names = [s.name for s in all_syms] if all_syms else []
    for s in WATCHLIST:
        if s in all_names and mt5.symbol_select(s, True):
            symbols_ok.append(s)
        else:
            print(f"[WARN] Symbol not available or select failed: {s}")
    if not symbols_ok:
        print("No symbols selected. Exiting.")
        mt5.shutdown()
        return

    acc = mt5.account_info()
    if not acc:
        print("Could not read account info.")
        mt5.shutdown()
        return
    start_equity = acc.equity
    print(f"Account: balance={acc.balance:.2f} equity={acc.equity:.2f}")

    state = RuntimeState(start_equity=start_equity)

    try:
        while True:
            state.loop += 1
            acc = mt5.account_info()
            if not acc:
                print("Account info missing; stopping.")
                break

            equity = acc.equity
            dd = (state.start_equity - equity) / state.start_equity
            gain = (equity - state.start_equity) / state.start_equity

            if state.loop % LOG_EVERY == 0:
                print(f"[{datetime.utcnow()}] EQ={equity:.2f} Gain={gain:.2%} DD={dd:.2%}")

            # global risk stops
            if dd >= MAX_DRAWDOWN_PCT:
                print("[KILL] Max drawdown hit. Closing all & stopping.")
                close_all()
                break

            # if gain >= DAILY_EQUITY_TARGET_PCT:
            #     print("[TARGET] Daily equity target reached. Closing all & stopping.")
            #     close_all()
            #     break

            # trailing updates
            for symbol in WATCHLIST:
                if not state.trail_due(symbol):
                    continue
                positions = mt5.positions_get(symbol=symbol)
                if positions:
                    # compute ATR for trailing
                    atr = atr_value(symbol, timeframe=ATR_TF, period=ATR_PERIOD)
                    if atr is None:
                        continue
                    for p in positions:
                        side = "buy" if p.type == mt5.POSITION_TYPE_BUY else "sell"
                        ask, bid = price_now(symbol)
                        if ask is None: continue
                        price = bid if side == "buy" else ask  # for SL below (buy) use bid, for sell use ask
                        trail_dist = atr * TRAIL_ATR_MULT
                        new_sl = (price - trail_dist) if side == "buy" else (price + trail_dist)
                        # only move SL in the direction of profit
                        if p.sl == 0.0:
                            modify_sl(symbol, p.ticket, new_sl)
                        else:
                            if (side == "buy" and new_sl > p.sl) or (side == "sell" and new_sl < p.sl):
                                modify_sl(symbol, p.ticket, new_sl)
                state.mark_trail(symbol)

            # entries
            for symbol in WATCHLIST:
                # respect cooldown and position limits
                if not state.cooldown_ok(symbol):
                    continue
                if not can_open_new(symbol):
                    continue

                primary, reasons = smc_signal(symbol)
                if not primary:
                    continue

                trend = htf_trend(symbol)
                aligned = (trend is None) or (trend == primary)

                # accept if (two+ agree) handled in smc_signal OR (single strong + HTF align)
                votes = [v for v in reasons.values() if v is not None]
                two_agree = len(votes) >= 2 and len(set(votes)) == 1
                acceptable = two_agree or (aligned and primary is not None)

                if not acceptable:
                    continue

                # compute ATR, SL/TP, and lot
                atr = atr_value(symbol, timeframe=ATR_TF, period=ATR_PERIOD)
                if atr is None or atr <= 0:
                    continue

                sl, tp = compute_sl_tp(symbol, primary, atr)
                if sl is None or tp is None:
                    continue

                lot = risk_based_lot(symbol, acc.balance, atr)
                if lot <= 0:
                    continue

                res = place_order(symbol, primary, lot, sl=sl, tp=tp)
                if res and getattr(res, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                    state.mark_trade(symbol)

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Bot crashed: {e}")
    finally:
        mt5.shutdown()
        print("MT5 shutdown.")

# ====== ENTRYPOINT ======
if __name__ == "__main__":
    run()
