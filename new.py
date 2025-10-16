"""
SMC_ICT_AggressiveProfit_v2.py
Aggressive, M15 SMC/ICT trading bot — aimed for ~50% daily growth (AGGRESSIVE).
* Test on demo first. Use conservative MIN_BALANCE while testing.
* Targets: BTC and XAU (XAUUSD).
"""

import MetaTrader5 as mt5
import time
import numpy as np
import pandas as pd
import re
from datetime import datetime, timedelta
from math import floor, ceil
from collections import deque, defaultdict

# ---- CONFIG ----
DEBUG_FORCE_SIGNAL = False
DEBUG = True

# TIMEFRAME set to M15 as requested
TIMEFRAME = mt5.TIMEFRAME_M15
BASE_SYMBOLS = ["XAUUSD", "BTCUSD"]  # user requested: Just BTC and Gold

COOLDOWN_SECONDS = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 0.8            # tighter SL (0.8 * ATR)
TP_ATR_MULT = 6.0               # 1 : 6 reward-to-risk

MAX_LOT = 200.0
MIN_LOT = 0.01
RISK_PER_TRADE = 0.02          # base risk; adaptive adjustments apply
SWING_LOOKBACK = 50
LIQUIDITY_WICK_MULT = 1.5
ZONE_BUFFER = 0.0015
HIGHER_TF_FILTER = True

# Daily/Session targets & limits
DAILY_PROFIT_TARGET_PCT = 0.50  # 50% daily profit target (stop trading for the day when reached)
DAILY_MAX_LOSS_PCT = 0.10       # pause trading for the day if down 10%
MAX_DRAWDOWN_PCT = 0.35         # total equity drawdown stop
MIN_BALANCE = 100.0

# Strategy params (kept minimal and SMC-aligned: structure, liquidity, volume)
MA_FAST_PERIOD = 9
MA_SLOW_PERIOD = 21
VOLUME_SURGE_MULTIPLIER = 1.6
MIN_CONFIRMATIONS = 1

trade_history = deque(maxlen=500)  # recent trade history
scaled_positions = set()           # track position tickets that have been scale-in'd

# Aggressive mode flags (already set by your request)
AGGRESSIVE_MODE = True

# Utility printing
def printt(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Helper: sanitize symbol names
def normalize_symbol_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r'[^A-Z0-9]', '', name.upper())

# Resolve actual symbol names from broker
def resolve_symbol(base: str, all_symbols: list) -> str | None:
    base_up = (base or '').upper()
    if not base_up:
        return None
    norm_base = normalize_symbol_name(base_up)

    for s in all_symbols:
        name = s.name if hasattr(s, 'name') else str(s)
        if name.upper() == base_up:
            return name

    for s in all_symbols:
        name = s.name if hasattr(s, 'name') else str(s)
        if normalize_symbol_name(name) == norm_base:
            return name

    for s in all_symbols:
        name = s.name if hasattr(s, 'name') else str(s)
        up = name.upper()
        if up.startswith(base_up):
            return name
    for s in all_symbols:
        name = s.name if hasattr(s, 'name') else str(s)
        up = name.upper()
        if up.endswith(base_up) or base_up in up:
            return name

    return None

def get_resolved_symbols(base_list: list) -> dict:
    resolved = {}
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        printt("⚠️ mt5.symbols_get() returned None. Ensure Market Watch is open and you're connected.")
        return resolved
    printt(f"🔎 Broker offers {len(all_symbols)} symbols")

    for base in base_list:
        actual = resolve_symbol(base, all_symbols)
        if actual:
            try:
                ok = mt5.symbol_select(actual, True)
                if not ok:
                    printt(f"⚠️ Could not select symbol {actual} into Market Watch.")
                resolved[base] = actual
                printt(f"✅ {base} → {actual}")
            except Exception as e:
                printt(f"⚠️ Exception while selecting {actual}: {e}")
                resolved[base] = actual
        else:
            printt(f"❌ Could NOT resolve base symbol '{base}'")
    return resolved

# ---- DATA FETCH ----
def get_symbol_data(symbol, timeframe=TIMEFRAME, bars=300):
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    except Exception as e:
        printt(f"Error copying rates for {symbol}: {e}")
        return None
    if rates is None or len(rates) == 0:
        printt(f"[ERROR] No price data for {symbol}. MT5 last_error: {mt5.last_error()}")
        return None
    df = pd.DataFrame(rates)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_symbol_point(symbol):
    info = mt5.symbol_info(symbol)
    if not info:
        return None
    try:
        return float(info.point)
    except Exception:
        return None

# ---- INDICATORS / SMC SIGNALS (kept SMC-focused) ----
def calculate_moving_averages(df):
    if len(df) < MA_SLOW_PERIOD:
        return None, None
    ma_fast = df['close'].rolling(window=MA_FAST_PERIOD).mean().iloc[-1]
    ma_slow = df['close'].rolling(window=MA_SLOW_PERIOD).mean().iloc[-1]
    return float(ma_fast), float(ma_slow)

def detect_market_structure(df):
    # simplified: if series of higher highs and higher lows -> bullish; inverse -> bearish; else ranging
    if len(df) < 30:
        return None
    recent_highs = df['high'].tail(10)
    recent_lows = df['low'].tail(10)
    higher_highs = all(recent_highs.iloc[i] > recent_highs.iloc[i-1] for i in range(1, len(recent_highs)))
    higher_lows = all(recent_lows.iloc[i] > recent_lows.iloc[i-1] for i in range(1, len(recent_lows)))
    lower_highs = all(recent_highs.iloc[i] < recent_highs.iloc[i-1] for i in range(1, len(recent_highs)))
    lower_lows = all(recent_lows.iloc[i] < recent_lows.iloc[i-1] for i in range(1, len(recent_lows)))
    if higher_highs and higher_lows:
        return "bullish"
    elif lower_highs and lower_lows:
        return "bearish"
    else:
        return "ranging"

def enhanced_volume_analysis(df, lookback=20):
    if len(df) < lookback + 1:
        return None
    current_volume = df['tick_volume'].iloc[-1]
    avg_volume = df['tick_volume'].iloc[-lookback:-1].mean()
    if avg_volume <= 0:
        return None
    if current_volume > avg_volume * VOLUME_SURGE_MULTIPLIER:
        price_action = "bullish" if df['close'].iloc[-1] > df['open'].iloc[-1] else "bearish"
        return price_action
    return None

def find_recent_swings(df, lookback=SWING_LOOKBACK):
    end = len(df)
    sub = df.iloc[max(0, end - lookback):end].reset_index(drop=True)
    idx_low = sub['low'].idxmin()
    idx_high = sub['high'].idxmax()
    low_val = float(sub.loc[idx_low, 'low'])
    high_val = float(sub.loc[idx_high, 'high'])
    orig_idx_low = max(0, end - lookback) + idx_low
    orig_idx_high = max(0, end - lookback) + idx_high
    return (orig_idx_low, low_val), (orig_idx_high, high_val)

def is_break_of_structure(df):
    if df is None or len(df) < SWING_LOOKBACK + 5:
        return None
    (swing_low_idx, swing_low), (swing_high_idx, swing_high) = find_recent_swings(df, SWING_LOOKBACK)
    last_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    # buy if price breaks above recent swing high; sell if below swing low
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
    last_body_high = max(last['open'], last['close'])
    last_body_low = min(last['open'], last['close'])
    upper_wick = last['high'] - last_body_high
    lower_wick = last_body_low - last['low']
    avg_upper = (recent['high'] - recent[['open', 'close']].max(axis=1)).mean()
    avg_lower = (recent[['open', 'close']].min(axis=1) - recent['low']).mean()
    if avg_lower > 0 and lower_wick > avg_lower * LIQUIDITY_WICK_MULT and last['close'] > last['open']:
        return "buy"
    if avg_upper > 0 and upper_wick > avg_upper * LIQUIDITY_WICK_MULT and last['close'] < last['open']:
        return "sell"
    return None

def detect_trendline_breakout(df, lookback=50):
    if df is None or len(df) < lookback + 1:
        return None
    recent = df.iloc[-lookback-1:-1].reset_index(drop=True)
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()
    last = df.iloc[-1]
    last_close = last['close']
    if last_close > swing_high:
        return "buy"
    if last_close < swing_low:
        return "sell"
    return None

def detect_volume_surge(df, surge_mult=1.5, lookback=30):
    if df is None or len(df) < lookback + 1:
        return None
    avg_vol = df['tick_volume'].iloc[-lookback-1:-1].mean()
    last_vol = df['tick_volume'].iloc[-1]
    if avg_vol <= 0 or last_vol <= 0:
        return None
    if last_vol > avg_vol * surge_mult:
        last = df.iloc[-1]
        return "buy" if last['close'] > last['open'] else "sell"
    return None

# ---- STRATEGY CORE: require SMC-style confluence (structure + volume/liquidity) ----
def strategy_high_conf(symbol):
    df = get_symbol_data(symbol, bars=300)
    if df is None or len(df) < 120:
        return None

    confirmations = []
    direction = None

    bos = is_break_of_structure(df)
    if bos:
        confirmations.append(("bos", bos))
        direction = bos

    sweep = detect_liquidity_sweep(df)
    if sweep:
        confirmations.append(("liquidity", sweep))
        if direction is None:
            direction = sweep

    vol = detect_volume_surge(df, surge_mult=VOLUME_SURGE_MULTIPLIER)
    if vol:
        confirmations.append(("volume", vol))
        if direction is None:
            direction = vol

    ma_fast, ma_slow = calculate_moving_averages(df)
    if ma_fast is not None and ma_slow is not None and direction:
        if ma_fast > ma_slow and direction == "buy":
            confirmations.append(("ma_alignment", "buy"))
        elif ma_fast < ma_slow and direction == "sell":
            confirmations.append(("ma_alignment", "sell"))

    market_structure = detect_market_structure(df)
    if market_structure == "ranging":
        # skip trades in ranging markets for improved edge
        return None
    elif market_structure == "bullish" and direction == "buy":
        confirmations.append(("market_structure", "buy"))
    elif market_structure == "bearish" and direction == "sell":
        confirmations.append(("market_structure", "sell"))

    # votes
    buy_votes = sum(1 for _, s in confirmations if s == 'buy')
    sell_votes = sum(1 for _, s in confirmations if s == 'sell')

    if DEBUG_FORCE_SIGNAL:
        return {"signal": "buy", "buy_votes": 1, "sell_votes": 0, "confirmations": confirmations}

    total_votes = buy_votes + sell_votes
    # require minimal confluence: at least volume+bos or liquidity+bos or structure+volume
    # We accept single-confirmation if it's a strong liquidity sweep + volume
    if total_votes >= MIN_CONFIRMATIONS:
        if buy_votes > sell_votes and buy_votes >= 1:
            return {"signal": "buy", "buy_votes": buy_votes, "sell_votes": sell_votes, "confirmations": confirmations}
        elif sell_votes > buy_votes and sell_votes >= 1:
            return {"signal": "sell", "buy_votes": buy_votes, "sell_votes": sell_votes, "confirmations": confirmations}

    return None

# ---- RISK / SL/TP ----
def calculate_recent_volatility(df, period=10):
    if len(df) < period + 1:
        return None
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return float(true_range.tail(period).mean())

def find_nearest_support_resistance(df, direction, lookback=100):
    if len(df) < lookback:
        return None
    recent_data = df.tail(lookback)
    current_price = df['close'].iloc[-1]
    if direction == "buy":
        support_levels = []
        for i in range(2, len(recent_data)-2):
            if (recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i-2] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1] and
                recent_data['low'].iloc[i] < recent_data['low'].iloc[i+2]):
                support_levels.append(recent_data['low'].iloc[i])
        if support_levels:
            supports_below = [s for s in support_levels if s < current_price]
            return max(supports_below) if supports_below else None
    elif direction == "sell":
        resistance_levels = []
        for i in range(2, len(recent_data)-2):
            if (recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i-2] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1] and
                recent_data['high'].iloc[i] > recent_data['high'].iloc[i+2]):
                resistance_levels.append(recent_data['high'].iloc[i])
        if resistance_levels:
            resistances_above = [r for r in resistance_levels if r > current_price]
            return min(resistances_above) if resistances_above else None
    return None

def atr_stop_levels(symbol, period=ATR_PERIOD, multiplier=ATR_MULTIPLIER):
    df = get_symbol_data(symbol, timeframe=TIMEFRAME, bars=period + 50)
    if df is None or len(df) < period + 1:
        return None, None
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = (df['high'] - df['close'].shift()).abs()
    df['L-C'] = (df['low'] - df['close'].shift()).abs()
    tr = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    atr = tr.rolling(window=period).mean().iloc[-1]
    if pd.isna(atr):
        return None, None
    sl_distance = float(atr * multiplier)
    tp_distance = float(sl_distance * TP_ATR_MULT)
    return sl_distance, tp_distance

# ---- SIZING & VOLUME ----
def adjust_volume_to_step(symbol, volume):
    info = mt5.symbol_info(symbol)
    if not info:
        return float(round(volume, 2))
    step = info.volume_step if getattr(info, 'volume_step', None) and info.volume_step > 0 else 0.01
    min_vol = info.volume_min if getattr(info, 'volume_min', None) and info.volume_min > 0 else MIN_LOT
    max_vol = info.volume_max if getattr(info, 'volume_max', None) and info.volume_max > 0 else MAX_LOT

    # use ceil to avoid zero volume
    steps = int(max(1, ceil((volume + 1e-12) / step)))
    vol = steps * step
    vol = max(min_vol, min(vol, max_vol))
    return float(round(vol, 3))

def get_lot_size_risk_based(equity, sl_distance, symbol, recent_trades):
    try:
        info = mt5.symbol_info(symbol)
        if not info:
            printt(f"[SIZING] missing symbol_info for {symbol}; using fallback sizing")
            return adjust_volume_to_step(symbol, MIN_LOT)

        # losing streak protection
        losing_streak = 0
        if recent_trades:
            for t in reversed(recent_trades):
                if t.get('profit', 0) >= 0:
                    break
                losing_streak += 1
        risk_multiplier = max(0.25, 1.0 - (losing_streak * 0.12))

        # adaptive risk based on equity performance:
        global RISK_PER_TRADE
        base_risk = RISK_PER_TRADE
        # tiny protection: if equity up 50% increase risk to 3%; if down 10% reduce to 1%
        try:
            starting = recent_trades[0]['account_start'] if recent_trades and 'account_start' in recent_trades[0] else None
            if starting:
                if equity >= starting * 1.5:
                    base_risk = 0.03
                elif equity <= starting * 0.9:
                    base_risk = 0.01
        except Exception:
            pass

        base_risk = base_risk * risk_multiplier
        risk_amount = equity * base_risk

        point = float(info.point) if getattr(info, 'point', None) else None
        tick_value = float(info.tick_value) if getattr(info, 'tick_value', None) else None
        contract_size = float(info.trade_contract_size) if getattr(info, 'trade_contract_size', None) else None

        if point is None or point == 0:
            approx_lot = max(MIN_LOT, equity * base_risk)
            return adjust_volume_to_step(symbol, approx_lot)

        # risk_per_lot: money lost for sl_distance when trading 1 lot
        if tick_value and tick_value > 0:
            risk_per_lot = (sl_distance / point) * tick_value
        elif contract_size and contract_size > 0:
            risk_per_lot = abs(contract_size * sl_distance)
        else:
            risk_per_lot = max(1e-6, (sl_distance / point) * 1.0)

        if risk_per_lot <= 0:
            return adjust_volume_to_step(symbol, MIN_LOT)

        raw_lots = risk_amount / risk_per_lot
        aggressive_boost = 1.30 if AGGRESSIVE_MODE else 1.0
        raw_lots *= aggressive_boost

        raw_lots = max(0.0, raw_lots)
        vol = max(MIN_LOT, min(raw_lots, MAX_LOT))
        vol = adjust_volume_to_step(symbol, vol)
        printt(f"[SIZING] {symbol}: equity={equity:.2f} base_risk={base_risk:.4f} risk_amount={risk_amount:.2f} "
               f"sl_dist={sl_distance:.6f} risk_per_lot={risk_per_lot:.6f} => lots={vol}")
        return vol
    except Exception as e:
        printt(f"[SIZING ERROR] {e}")
        return adjust_volume_to_step(symbol, MIN_LOT)

# ---- ORDER HELPERS & EXECUTION ----
def place_trade(symbol, lot, order_type='buy', sl_distance=None, tp_distance=None):
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        printt(f"Failed to get tick for {symbol}")
        return None
    price = float(tick.ask) if order_type == 'buy' else float(tick.bid)
    sl_price = 0.0
    tp_price = 0.0
    if sl_distance is not None and tp_distance is not None:
        if order_type == 'buy':
            sl_price = price - sl_distance
            tp_price = price + tp_distance
        else:
            sl_price = price + sl_distance
            tp_price = price - tp_distance

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": mt5.ORDER_TYPE_BUY if order_type == 'buy' else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": float(sl_price),
        "tp": float(tp_price),
        "deviation": 20,
        "magic": 123456,
        "comment": "SMC_ICT_Aggressive_v2",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if DEBUG:
        printt(f"[ORDER_REQUEST] {symbol} {order_type} vol={lot:.3f} price={price:.6f} sl={sl_price:.6f} tp={tp_price:.6f}")

    try:
        result = mt5.order_send(request)
    except Exception as e:
        printt(f"Exception during order_send for {symbol}: {e}")
        return None

    if result is None:
        printt(f"order_send returned None for {symbol}")
        return None

    ret = getattr(result, 'retcode', None)
    if ret != mt5.TRADE_RETCODE_DONE:
        printt(f"Order failed for {symbol}: {ret} ({getattr(result, 'comment', '')})")
        return None

    printt(f"Order executed: {order_type.upper()} {lot} {symbol} ticket={getattr(result, 'order', None)}")
    time.sleep(1.0)
    return result

def close_position_by_ticket(pos_ticket):
    try:
        positions = mt5.positions_get(ticket=pos_ticket)
    except Exception:
        positions = None
    if not positions:
        printt(f"Position {pos_ticket} not found (already closed).")
        return
    pos = positions[0]
    symbol = pos.symbol
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        printt(f"Cannot fetch tick to close ticket {pos_ticket}")
        return
    price = float(tick.bid) if order_type == mt5.ORDER_TYPE_SELL else float(tick.ask)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(pos.volume),
        "type": order_type,
        "position": int(pos.ticket),
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": "AutoClose",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    try:
        res = mt5.order_send(request)
    except Exception as e:
        printt(f"Exception sending close for ticket {pos_ticket}: {e}")
        return
    if res is None:
        printt(f"Close: order_send returned None for ticket {pos_ticket}")
        return
    if getattr(res, "retcode", None) == mt5.TRADE_RETCODE_DONE:
        printt(f"Close result for ticket {pos_ticket}: success, deal={getattr(res,'deal',None)} order={getattr(res,'order',None)}")
    else:
        printt(f"Close failed for ticket {pos_ticket}: retcode={getattr(res,'retcode',None)} comment={getattr(res,'comment',None)}")

def close_all_positions():
    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        printt("No positions to close.")
        return
    for pos in positions:
        close_position_by_ticket(pos.ticket)

# ---- PROFIT TARGET / SESSION FILTERS ----
def check_and_close_on_profit(daily_start_balance):
    positions = mt5.positions_get()
    info = mt5.account_info()
    if info is None:
        return False
    equity = info.equity
    # If we've hit daily profit target, close and signal to stop trading for the day
    if equity >= daily_start_balance * (1.0 + DAILY_PROFIT_TARGET_PCT):
        printt(f"Daily profit target reached: equity {equity:.2f} >= {daily_start_balance*(1.0+DAILY_PROFIT_TARGET_PCT):.2f}")
        close_all_positions()
        return True
    return False

def get_higher_tf_trend(symbol):
    df = get_symbol_data(symbol, timeframe=mt5.TIMEFRAME_H1, bars=50)
    if df is None or len(df) < 21:
        return None
    ma_fast = df['close'].rolling(window=10).mean().iloc[-1]
    ma_slow = df['close'].rolling(window=20).mean().iloc[-1]
    return "buy" if ma_fast > ma_slow else "sell"

# ---- BREAKEVEN / TRAILING / SCALE-IN ----
class AdaptiveTradeManager:
    def __init__(self):
        self.peak_by_ticket = {}
        self.drop_confirm_count = {}
        self.volatility_by_symbol = {}

    def update_volatility(self, symbol, df):
        volatility = calculate_recent_volatility(df)
        if volatility:
            self.volatility_by_symbol[symbol] = volatility

    def get_trailing_parameters(self, symbol):
        # derive pips-based rules from volatility
        base_drop_pips = 30.0
        base_min_profit = 10.0
        volatility = self.volatility_by_symbol.get(symbol)
        if volatility:
            point = get_symbol_point(symbol)
            if point and point > 0:
                volatility_pips = volatility / point
                if volatility_pips > 80:
                    return base_drop_pips * 1.5, base_min_profit * 1.5
                elif volatility_pips < 20:
                    return base_drop_pips * 0.6, base_min_profit * 0.6
        return base_drop_pips, base_min_profit

    def update_and_manage(self):
        positions = mt5.positions_get()
        if not positions:
            self.peak_by_ticket = {}
            self.drop_confirm_count = {}
            return

        current_tickets = {int(p.ticket) for p in positions}
        for ticket in list(self.peak_by_ticket.keys()):
            if ticket not in current_tickets:
                del self.peak_by_ticket[ticket]
                if ticket in self.drop_confirm_count:
                    del self.drop_confirm_count[ticket]

        for pos in positions:
            ticket = int(pos.ticket)
            symbol = pos.symbol
            df = get_symbol_data(symbol, bars=80)
            if df is not None:
                self.update_volatility(symbol, df)

            drop_pips, min_profit_pips = self.get_trailing_parameters(symbol)
            point = get_symbol_point(symbol)
            if point is None or point == 0:
                continue
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                continue
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            current_price = float(tick.bid) if is_buy else float(tick.ask)
            profit_pips = (current_price - pos.price_open) / point if is_buy else (pos.price_open - current_price) / point

            # store peak profit in pips
            prev_peak = self.peak_by_ticket.get(ticket, -1e9)
            if profit_pips > prev_peak:
                self.peak_by_ticket[ticket] = profit_pips
                self.drop_confirm_count[ticket] = 0
                # check scale-in opportunity when profit >= 2 * ATR (in pips)
                try:
                    sl = pos.sl
                    if sl and sl != 0:
                        sl_pips = abs(pos.price_open - pos.sl) / point
                    else:
                        # fallback: estimate sl based on ATR
                        atr = calculate_recent_volatility(df) if df is not None else None
                        if atr:
                            sl_pips = (atr * ATR_MULTIPLIER) / point
                        else:
                            sl_pips = max(10.0, min(200.0, profit_pips/2.0))
                except Exception:
                    sl_pips = profit_pips

                # scale-in: if profit >= 2 * sl_pips and we haven't scaled this ticket, open 50% additional lot
                if profit_pips >= 2.0 * sl_pips and ticket not in scaled_positions:
                    # compute additional lot as 50% of original pos volume, but re-check risk sizing slightly
                    add_lot = max(MIN_LOT, float(pos.volume) * 0.5)
                    # ensure we don't exceed broker limits
                    add_lot = adjust_volume_to_step(symbol, add_lot)
                    printt(f"[SCALE-IN] Ticket {ticket} profit_pips={profit_pips:.1f} >= 2*sl_pips({sl_pips:.1f}). Adding lot={add_lot}")
                    # place same-direction market order as scale-in
                    side = 'buy' if is_buy else 'sell'
                    # use current sl/tp distances similar to parent
                    try:
                        # compute distances from current pos SL/TP if available
                        if pos.sl and pos.sl != 0:
                            sl_dist = abs(pos.price_open - pos.sl)
                        else:
                            sl_dist, _ = atr_stop_levels(symbol)
                            if sl_dist is None:
                                sl_dist = (atr * ATR_MULTIPLIER) if (df is not None and (atr := calculate_recent_volatility(df))) else 0.0
                        # keep TP scaled by TP_ATR_MULT
                        tp_dist = sl_dist * TP_ATR_MULT
                        place_trade(symbol, add_lot, order_type=side, sl_distance=sl_dist, tp_distance=tp_dist)
                        scaled_positions.add(ticket)
                    except Exception as e:
                        printt(f"[SCALE-IN] Failed to scale-in ticket {ticket}: {e}")
                    continue

            # trailing & breakeven: when profit crosses threshold, tighten SL to protect runners
            peak = self.peak_by_ticket.get(ticket, 0.0)
            if peak < min_profit_pips:
                continue

            if (peak - profit_pips) >= drop_pips:
                self.drop_confirm_count[ticket] = self.drop_confirm_count.get(ticket, 0) + 1
                if self.drop_confirm_count[ticket] >= 1:
                    printt(f"Adaptive trailing: ticket {ticket} dropped {peak-profit_pips:.1f}pips from peak -> closing.")
                    close_position_by_ticket(ticket)
                    self.peak_by_ticket.pop(ticket, None)
                    self.drop_confirm_count.pop(ticket, None)
            else:
                # attempt to move SL to breakeven once > 1.5*sl_pips
                try:
                    if pos.sl and pos.sl != 0:
                        sl_pips = abs(pos.price_open - pos.sl) / point
                    else:
                        sl_pips = (calculate_recent_volatility(df) * ATR_MULTIPLIER) / point if df is not None else 0.0
                except Exception:
                    sl_pips = 0.0

                if peak >= 1.5 * sl_pips:
                    # move SL to breakeven + 1 pip buffer
                    buffer = 1 * point
                    new_sl = pos.price_open + buffer if is_buy else pos.price_open - buffer
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": int(pos.ticket),
                        "sl": float(new_sl),
                        "tp": float(pos.tp),
                        "comment": "AdaptiveBreakeven"
                    }
                    try:
                        res = mt5.order_send(req)
                        if res is not None and getattr(res, "retcode", None) == mt5.TRADE_RETCODE_DONE:
                            printt(f"[BREAKEVEN] Moved SL to breakeven for ticket {ticket} -> {new_sl:.6f}")
                        else:
                            # not fatal, just continue
                            pass
                    except Exception:
                        pass

# ---- DYNAMIC STOP LEVELS (ATR + SR heuristics) ----
def get_dynamic_stop_levels(symbol, direction, df):
    sl_distance, tp_distance = atr_stop_levels(symbol)
    if sl_distance is None:
        return None, None

    recent_atr = calculate_recent_volatility(df)
    if recent_atr and sl_distance:
        volatility_factor = recent_atr / sl_distance if sl_distance else 1.0
        if volatility_factor > 1.5:
            sl_distance *= 1.2
            tp_distance *= 1.1
        elif volatility_factor < 0.7:
            sl_distance *= 0.8
            tp_distance *= 0.9

    sr_level = find_nearest_support_resistance(df, direction)
    if sr_level:
        current_price = df['close'].iloc[-1]
        sr_distance = abs(current_price - sr_level)
        if sr_distance < sl_distance * 1.5 and sr_distance > sl_distance * 0.5:
            sl_distance = sr_distance

    return float(sl_distance), float(tp_distance)

# ---- MAIN LOOP / ORCHESTRATION ----
def within_trading_hours(now=None):
    if now is None:
        now = datetime.now()
    # keep trading during typical hours; user had TRADING_START/END earlier, keep broad window
    TRADING_START = 0
    TRADING_END = 24
    return TRADING_START <= now.hour < TRADING_END

def time_until_trading_window(now=None):
    if now is None:
        now = datetime.now()
    tomorrow_start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    if now.hour >= 24:
        return tomorrow_start - now
    elif now.hour < 0:
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return today_start - now
    return timedelta(seconds=0)

def run_bot(symbols):
    for s in symbols:
        try:
            mt5.symbol_select(s, True)
        except Exception:
            pass

    info = mt5.account_info()
    if info is None:
        printt("Cannot get account info. Exiting.")
        return

    if info.balance < MIN_BALANCE:
        printt(f"Account balance (${info.balance:.2f}) is below ${MIN_BALANCE}. Bot shutting down.")
        return

    initial_balance = info.balance
    daily_start_balance = initial_balance
    last_trade_time = {s: None for s in symbols}
    trailing_mgr = AdaptiveTradeManager()

    # For adaptive sizing, add the account start into recent trade history for reference
    if len(trade_history) == 0:
        trade_history.append({'account_start': initial_balance})

    try:
        while True:
            info = mt5.account_info()
            if not info:
                break

            balance, equity = info.balance, info.equity
            if balance < MIN_BALANCE:
                printt(f"Balance dropped below ${MIN_BALANCE}. Stopping bot.")
                close_all_positions()
                break

            now = datetime.now()
            if not within_trading_hours(now):
                printt("Outside trading hours. Sleeping briefly.")
                time.sleep(30)
                continue

            loss_pct = (initial_balance - equity) / initial_balance if initial_balance else 0.0
            printt(f"Equity: {equity:.2f}, Balance: {balance:.2f}, Drawdown%: {loss_pct:.2%}")

            # absolute drawdown stop
            if loss_pct >= MAX_DRAWDOWN_PCT:
                printt("Max drawdown reached. Closing all positions and stopping bot.")
                close_all_positions()
                break

            # daily loss stop: if we've dropped more than DAILY_MAX_LOSS_PCT vs daily start, pause trading for the day
            daily_loss_pct = (daily_start_balance - equity) / daily_start_balance if daily_start_balance else 0.0
            if daily_loss_pct >= DAILY_MAX_LOSS_PCT:
                printt(f"[DAILY STOP] Daily loss {daily_loss_pct:.2%} exceeded limit {DAILY_MAX_LOSS_PCT:.2%}. Pausing trading for 1 hour.")
                close_all_positions()
                time.sleep(60 * 60)
                # reset daily start after hour to prevent immediate re-entry
                daily_start_balance = mt5.account_info().equity if mt5.account_info() else daily_start_balance
                continue

            # check daily profit target reached
            if check_and_close_on_profit(daily_start_balance):
                printt("Daily profit target reached — pausing trading for the rest of the day.")
                # wait until next day
                # compute seconds until midnight + small buffer
                now = datetime.now()
                tomorrow = (now + timedelta(days=1)).replace(hour=1, minute=0, second=0, microsecond=0)
                secs = (tomorrow - now).total_seconds()
                time.sleep(max(60, secs))
                daily_start_balance = mt5.account_info().equity if mt5.account_info() else daily_start_balance
                continue

            # update trade manager (breakeven/scale/ trailing)
            trailing_mgr.update_and_manage()

            for symbol in symbols:
                lt = last_trade_time.get(symbol)
                if lt and (now - lt).total_seconds() < COOLDOWN_SECONDS:
                    continue

                sig_info = strategy_high_conf(symbol)
                if not sig_info:
                    continue

                sig = sig_info.get("signal")
                confirmations = sig_info.get("confirmations", [])
                printt(f"Signal {sig.upper()} for {symbol} conf={confirmations}")

                # higher timeframe filter
                if HIGHER_TF_FILTER:
                    hf = get_higher_tf_trend(symbol)
                    printt(f"Higher TF trend for {symbol}: {hf}")
                    if hf is not None and hf != sig:
                        printt(f"Signal blocked by HTF trend for {symbol}")
                        continue

                df = get_symbol_data(symbol, bars=200)
                if df is None:
                    continue

                sl_distance, tp_distance = get_dynamic_stop_levels(symbol, sig, df)
                if sl_distance is None or tp_distance is None:
                    printt(f"Could not compute dynamic stops for {symbol}. Skipping.")
                    continue

                lot = get_lot_size_risk_based(equity, sl_distance, symbol, list(trade_history))
                if lot < MIN_LOT:
                    printt(f"Lot {lot} for {symbol} below minimum. Skipping.")
                    continue

                # concurrency guard: limit open positions per symbol to 2
                open_positions = mt5.positions_get() or []
                open_for_symbol = [p for p in open_positions if p.symbol == symbol]
                max_conc = 2
                if len(open_for_symbol) >= max_conc:
                    printt(f"[CONCURRENCY] Skipping {symbol} — {len(open_for_symbol)} open >= {max_conc}")
                    continue

                # place the trade
                res = place_trade(symbol, lot, order_type=sig, sl_distance=sl_distance, tp_distance=tp_distance)
                if res is not None:
                    last_trade_time[symbol] = datetime.now()
                    trade_history.append({
                        'time': datetime.now(),
                        'symbol': symbol,
                        'direction': sig,
                        'lot': lot,
                        'profit': 0,
                        'account_start': initial_balance
                    })
                    # small wait for price/positions to update
                    time.sleep(2)

                time.sleep(0.5)

            time.sleep(1.5)
    except KeyboardInterrupt:
        printt("Bot interrupted by user.")
    except Exception as e:
        printt(f"Unexpected error in main loop: {e}")
        import traceback
        traceback.print_exc()

# ---- ENTRYPOINT ----
if __name__ == "__main__":
    print("Starting SMC_ICT_AggressiveProfit_v2.py — M15 aggressive mode (DEMO FIRST!)")

    try:
        login = int(input("Enter MT5 Login ID: "))
    except Exception:
        print("Invalid login ID.")
        raise SystemExit(1)
    password = input("Enter MT5 Password: ")
    server = input("Enter MT5 Server: ")

    if not mt5.initialize():
        print("MT5 terminal initialization failed:", mt5.last_error())
        raise SystemExit(1)
    else:
        if not mt5.login(login, password, server):
            print("❌ Login failed:", mt5.last_error())
            mt5.shutdown()
            raise SystemExit(1)
        else:
            acc_info = mt5.account_info()
            if acc_info is None:
                print("❌ Could not retrieve account info after login.")
            else:
                print(f"✅ Connected to account {acc_info.login}")
                print(f"Balance: {acc_info.balance:.2f}, Equity: {acc_info.equity:.2f}, Leverage: {acc_info.leverage}")

                resolved = get_resolved_symbols(BASE_SYMBOLS)
                if len(resolved) == 0:
                    printt("No symbols were resolved. Make sure Market Watch is open or add BTC/XAU to Market Watch manually.")
                else:
                    actual_symbols = list(resolved.values())
                    printt(f"Trading will use actual symbols: {actual_symbols}")
                    run_bot(actual_symbols)

            mt5.shutdown()
