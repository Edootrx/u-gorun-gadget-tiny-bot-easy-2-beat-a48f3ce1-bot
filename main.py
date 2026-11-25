"""Tiny Bot, Easy $2: Beat Buy-and-Hold on Solana.

Long-only paper-trading bot for agent-fun/bars-replay/v1, 7 hourly Solana
assets, $10,000 USDC. Python 3.13 standard library only, plus the kit's
vendored shim. Integer math throughout; no clock, no network, no randomness,
so two runs on the same bars give the same orders.

Three sleeves share the cash, one position per symbol, 20% of equity each
(under the 25% weight cap), every order max_slippage_bps 200:

1. Dip-buy (the edge in a falling market): an asset at least 8% below its
   highest close of the last 24 bars is bought for the bounce; exit at
   +3% from the entry close, or on a 3% trailing stop from the peak close.
2. Rotation (the edge in a rising market): each bar rank the assets by
   24-bar momentum and hold the top 3 whose momentum is positive but not
   parabolic (at most +3%, so a V-bounce top is not chased). A rotation
   position is sold when its momentum falls below -3% or its rank drops
   out of the top 3. Rotation is what keeps the bot invested while the
   market rises, and in cash while it falls.
3. Scalp (the trade-count floor): an asset at least 1% below its highest
   close of the last 12 bars is bought small (4% of equity) and sold at
   +1.5%, at -1.5%, or after 12 bars. Short-horizon mean reversion is
   roughly breakeven after fees and keeps the bot trading in quiet
   windows where the other two sleeves see nothing.

After a sell, a symbol rests 12 bars before a rotation or scalp
re-buy, so a stop-out is not re-bought into the same chop. Dip buys
are exempt: a fresh 8% drop is the entry itself, and delaying it cost
the bounce in testing. Orders decided on the last batch are rejected
by the replay, so late entries simply do not fill; positions left at
the end are flattened by the replay at the last close, as the rules
say. A symbol is bought at most once per batch.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
import makefun_bars

BPS = 10_000

SYMBOLS = ("FARTCOIN", "JITOSOL", "POPCAT", "RAY", "SOL", "WETH", "WIF")

# Rotation: 24-bar momentum, top 3, 20% of equity per position.
FAST = 8
SLOW = 24
MOM_N = 24
TOP_K = 3
SIZE_BPS = 2000
MOM_CAP_BPS = 300
ROT_EXIT_BPS = -300

# Dip-buy: 8% below the 24-bar high; exit +3% or 3% trailing stop.
DIP_DROP_BPS = 800
DIP_LOOKBACK = 24
DIP_EXIT_BPS = 300
DIP_TRAIL_BPS = 300

# Scalp: 1% below the 12-bar high, 4% of equity; exit +1.5% / -1.5% / 12 bars.
SCALP_DROP_BPS = 100
SCALP_LOOKBACK = 12
SCALP_SIZE_BPS = 400
SCALP_EXIT_BPS = 150
SCALP_STOP_BPS = 150
SCALP_HOLD = 12

COOLDOWN = 12
MAX_SLIP = 200
MIN_NOTIONAL = 1_000_000

# Fixed point for the EMAs: prices are scaled by this factor.
SCALE = 1000

# Per-symbol close history, from the batches themselves.
closes = {s: [] for s in SYMBOLS}
ema_fast = {}
ema_slow = {}
# Position bookkeeping: how a position was opened, its entry close, its
# peak close since entry, the bar it was opened on, and the last sell bar.
origin = {}
entry_close = {}
peak = {}
entry_seq = {}
last_sell = {}


def _momentum(symbol):
    """24-bar return in bps, or None while the history is too short."""
    hist = closes[symbol]
    if len(hist) <= MOM_N:
        return None
    return (hist[-1] - hist[-1 - MOM_N]) * BPS // hist[-1 - MOM_N]


def _dip_drop(symbol):
    """Drop in bps from the highest close of the last DIP_LOOKBACK bars."""
    hist = closes[symbol]
    if len(hist) <= DIP_LOOKBACK:
        return None
    high = max(hist[-1 - DIP_LOOKBACK:-1])
    return (high - hist[-1]) * BPS // high


def _scalp_drop(symbol):
    """Drop in bps from the highest close of the last SCALP_LOOKBACK bars."""
    hist = closes[symbol]
    if len(hist) <= SCALP_LOOKBACK:
        return None
    high = max(hist[-1 - SCALP_LOOKBACK:-1])
    return (high - hist[-1]) * BPS // high


def _resting(symbol, seq):
    return seq - last_sell.get(symbol, -10 ** 9) < COOLDOWN


def on_batch(state, batch):
    # Record this bar's closes and update the EMAs (integer math, seeded
    # on the first close, alpha 2/(n+1)).
    for symbol in SYMBOLS:
        c = state.close(symbol)
        if c is None:
            continue
        closes[symbol].append(c)
        if symbol not in ema_fast:
            ema_fast[symbol] = c * SCALE
            ema_slow[symbol] = c * SCALE
        else:
            ema_fast[symbol] = (2 * c * SCALE + (FAST - 1) * ema_fast[symbol]) // (FAST + 1)
            ema_slow[symbol] = (2 * c * SCALE + (SLOW - 1) * ema_slow[symbol]) // (SLOW + 1)
    if state.warming_up:
        return []

    orders = []
    seq = state.seq

    # Momentum ranking for the rotation sleeve.
    scores = {s: _momentum(s) for s in SYMBOLS}
    ranked = sorted(
        (s for s in SYMBOLS if scores[s] is not None),
        key=lambda s: -scores[s],
    )
    target = {s for s in ranked[:TOP_K]
              if scores[s] is not None and 0 < scores[s] <= MOM_CAP_BPS}

    def sell(symbol):
        holding = state.positions.get(symbol)
        if holding is None or holding.qty == 0:
            return
        orders.append({"side": "sell", "symbol": symbol, "qty": holding.qty,
                       "max_slippage_bps": MAX_SLIP})
        origin.pop(symbol, None)
        entry_close.pop(symbol, None)
        peak.pop(symbol, None)
        entry_seq.pop(symbol, None)
        last_sell[symbol] = seq

    # Scalp exits: profit target, stop, or max hold.
    for symbol in list(state.positions):
        if origin.get(symbol) != "scalp":
            continue
        c = closes[symbol][-1]
        ec = entry_close.get(symbol)
        if ec is None:
            continue
        move = (c - ec) * BPS // ec
        if (move >= SCALP_EXIT_BPS or move <= -SCALP_STOP_BPS
                or seq - entry_seq.get(symbol, seq) >= SCALP_HOLD):
            sell(symbol)

    # Dip exits: profit target, or trailing stop from the peak close.
    for symbol in list(state.positions):
        if origin.get(symbol) != "dip":
            continue
        c = closes[symbol][-1]
        peak[symbol] = max(peak.get(symbol, 0), c)
        ec = entry_close.get(symbol)
        if ec is None:
            continue
        if (c * BPS < peak[symbol] * (BPS - DIP_TRAIL_BPS)
                or (c - ec) * BPS // ec >= DIP_EXIT_BPS):
            sell(symbol)

    # Rotation exits: momentum clearly negative, or rank out of the top 3.
    # (Dip positions are exempt; scalp positions are not.)
    for symbol in list(state.positions):
        if origin.get(symbol) == "dip":
            continue
        if symbol in target:
            continue
        if scores[symbol] is None:
            continue
        if scores[symbol] < ROT_EXIT_BPS or ranked.index(symbol) > TOP_K:
            sell(symbol)

    # Entries, dip first, then rotation, then scalp. One position per
    # symbol. Dip buys are never held back by the cooldown: a fresh 8%
    # drop is the entry itself, and delaying it by a scalp stop-out cost
    # the bounce in testing. Rotation and scalp re-buys rest 12 bars
    # after a sell, so a stop-out is not re-bought into the same chop.
    # A symbol bought once in this batch is not bought again.
    bought = set()
    cands = []  # (group, -score, symbol, kind)
    for symbol in SYMBOLS:
        if symbol in state.positions:
            continue
        drop = _dip_drop(symbol)
        if drop is not None and drop >= DIP_DROP_BPS:
            cands.append((0, -drop, symbol, "dip"))
    for symbol in ranked:
        if symbol in state.positions or symbol not in target:
            continue
        if _resting(symbol, seq):
            continue
        cands.append((1, -scores[symbol], symbol, "rot"))
    for symbol in SYMBOLS:
        if symbol in state.positions or _resting(symbol, seq):
            continue
        drop = _scalp_drop(symbol)
        if drop is not None and drop >= SCALP_DROP_BPS:
            cands.append((2, -drop, symbol, "scalp"))
    cands.sort()

    cash = state.cash
    for _group, _neg, symbol, kind in cands:
        if symbol in bought:
            continue
        size_bps = SCALP_SIZE_BPS if kind == "scalp" else SIZE_BPS
        size = min(size_bps * state.equity // BPS, state.weight_room(symbol), cash)
        if size < MIN_NOTIONAL:
            continue
        orders.append({"side": "buy", "symbol": symbol,
                       "notional_quote_units": size, "max_slippage_bps": MAX_SLIP})
        bought.add(symbol)
        origin[symbol] = kind
        entry_close[symbol] = closes[symbol][-1]
        peak[symbol] = closes[symbol][-1]
        entry_seq[symbol] = seq
        cash -= size
        if len(orders) >= 16:
            break
    return orders


makefun_bars.run(on_batch)
