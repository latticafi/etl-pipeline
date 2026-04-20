"""Polymarket trade canonicalization.

A Polymarket market has two tradeable tokens, YES (asset_ids[0]) and NO
(asset_ids[1]). Economically, a trade on the NO token is a trade on (1 - YES)
in the opposite direction. The WARHORSE training pipeline canonicalizes
everything to the YES perspective before computing features. The ETL must
do the same at ingest time so downstream consumers never recompute and
never see mixed frames.

This module is the single source of truth. WSS handlers, duckdb import,
and Phase 0 read paths all go through these functions.

Two entry points because raw inputs differ:

  - WSS events: give us `asset_id` (the actual token traded); we look up
    against the market's `asset_ids` to decide YES vs NO.
  - duckdb rows: give us `nonusdc_side` as a pre-resolved 'token1' or
    'token2' string (token1 == YES, token2 == NO in Polymarket convention).

Both produce identical CanonicalTrade output.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class CanonicalTrade:
    """A trade, canonicalized to the YES perspective.

    canonical_price and trade_sign are the values downstream features and
    training queries consume. Raw price and asset_id are preserved on the
    CH table for diagnostics but not used in feature computation.

    Invariants:
      - 0 < canonical_price < 1 (Polymarket enforces this on every fill)
      - trade_sign in {-1, +1}
      - dollar_volume > 0
    """

    condition_id: str
    asset_id: str
    is_yes: bool
    price: float            # raw price on asset_id
    canonical_price: float  # YES-frame price
    size: float
    dollar_volume: float    # price * size, raw frame (not canonical)
    side: str               # 'BUY' or 'SELL', raw taker direction
    trade_sign: int         # +1 or -1 in canonical frame
    timestamp: datetime
    transaction_hash: str


# ---------------------------------------------------------------------------
# Canonicalization primitives
# ---------------------------------------------------------------------------


def canonical_price_from(raw_price: float, is_yes: bool) -> float:
    """NO-token price flipped; YES-token price passthrough."""
    return raw_price if is_yes else 1.0 - raw_price


def trade_sign_from(is_yes: bool, side: str) -> int:
    """Canonical sign: +1 for net buying pressure on YES, -1 for net selling.

    Truth table:
      YES token, BUY  → buying YES          → +1
      YES token, SELL → selling YES         → -1
      NO  token, BUY  → buying NO = selling YES  → -1
      NO  token, SELL → selling NO = buying YES  → +1

    Equivalently: +1 iff (is_yes == (side == "BUY")).
    """
    side_norm = side.upper().strip()
    if side_norm not in ("BUY", "SELL"):
        raise ValueError(f"side must be BUY or SELL, got {side!r}")
    is_buy = side_norm == "BUY"
    return 1 if is_yes == is_buy else -1


# ---------------------------------------------------------------------------
# WSS entry point
# ---------------------------------------------------------------------------


def canonicalize_wss(
    *,
    condition_id: str,
    asset_id: str,
    asset_ids: tuple[str, str] | list[str],
    raw_price: float,
    size: float,
    side: str,
    timestamp: datetime,
    transaction_hash: str,
) -> CanonicalTrade:
    """Canonicalize a trade from a Polymarket WSS `last_trade_price` event.

    asset_ids is the market's [yes_token, no_token] pair from Gamma. We
    identify YES vs NO by direct string comparison against asset_ids[0].

    Raises ValueError if asset_id is neither token of the market (shouldn't
    happen in practice; would indicate a subscribe/routing bug).
    """
    yes_token = asset_ids[0]
    no_token = asset_ids[1]
    if asset_id == yes_token:
        is_yes = True
    elif asset_id == no_token:
        is_yes = False
    else:
        raise ValueError(
            f"asset_id {asset_id!r} matches neither YES ({yes_token!r}) "
            f"nor NO ({no_token!r}) of market {condition_id!r}"
        )

    cp = canonical_price_from(raw_price, is_yes)
    ts = trade_sign_from(is_yes, side)

    return CanonicalTrade(
        condition_id=condition_id,
        asset_id=asset_id,
        is_yes=is_yes,
        price=raw_price,
        canonical_price=cp,
        size=size,
        dollar_volume=raw_price * size,
        side=side.upper().strip(),
        trade_sign=ts,
        timestamp=timestamp,
        transaction_hash=transaction_hash,
    )


# ---------------------------------------------------------------------------
# duckdb / historical import entry point
# ---------------------------------------------------------------------------


def canonicalize_duckdb(
    *,
    condition_id: str,
    asset_ids: tuple[str, str] | list[str],
    nonusdc_side: str,           # 'token1' (YES) or 'token2' (NO)
    raw_price: float,
    usd_amount: float,
    token_amount: float,
    taker_direction: str,        # 'BUY' or 'SELL'
    timestamp: datetime,
    tx_hash: str,
) -> CanonicalTrade:
    """Canonicalize a trade from the duckdb historical scrape.

    duckdb's schema uses `nonusdc_side` ('token1'/'token2') as the token
    label rather than the on-chain asset_id. token1 is YES, token2 is NO.
    We reconstruct the actual asset_id from the market's asset_ids pair.

    `usd_amount` is the dollar_volume directly; we don't recompute from
    price*size because rounding in the original scrape might give a slightly
    different value and we want to match what was scraped.
    """
    side_label = nonusdc_side.lower().strip()
    if side_label == "token1":
        is_yes = True
        asset_id = asset_ids[0]
    elif side_label == "token2":
        is_yes = False
        asset_id = asset_ids[1]
    else:
        raise ValueError(
            f"nonusdc_side must be 'token1' or 'token2', got {nonusdc_side!r}"
        )

    cp = canonical_price_from(raw_price, is_yes)
    ts = trade_sign_from(is_yes, taker_direction)

    return CanonicalTrade(
        condition_id=condition_id,
        asset_id=asset_id,
        is_yes=is_yes,
        price=raw_price,
        canonical_price=cp,
        size=token_amount,
        dollar_volume=usd_amount,
        side=taker_direction.upper().strip(),
        trade_sign=ts,
        timestamp=timestamp,
        transaction_hash=tx_hash,
    )


# ---------------------------------------------------------------------------
# Vectorized canonicalization for bulk imports
# ---------------------------------------------------------------------------


def canonicalize_batch(
    *,
    nonusdc_side: "np.ndarray",      # array of 'token1'/'token2' strings
    raw_price: "np.ndarray",         # float64
    taker_direction: "np.ndarray",   # array of 'BUY'/'SELL' strings
    token1_asset_id: "np.ndarray",   # str, the YES token per row
    token2_asset_id: "np.ndarray",   # str, the NO token per row
) -> dict[str, "np.ndarray"]:
    """Vectorized canonicalization for bulk imports.

    Applies identical logic to canonicalize_duckdb() but over numpy arrays,
    avoiding per-row function call overhead at 10^9 scale. A per-row call
    takes microseconds; over 1B rows that's ~2.7 hours of Python overhead
    we don't need to pay.

    Caller must pre-filter out invalid rows (price out of (0,1), bad
    nonusdc_side, bad taker_direction) — this function assumes clean input
    and does not raise on individual bad rows.

    Returns dict of output arrays:
      - is_yes: bool
      - asset_id: str (the actual token on the trade)
      - canonical_price: float64
      - trade_sign: int8 (+1 / -1)

    The caller is responsible for assembling these back with the other
    passthrough columns (timestamp, dollar_volume, size, etc.).
    """
    import numpy as np  # local import to keep per-row imports zero-cost

    is_yes = nonusdc_side == "token1"
    canonical_price = np.where(is_yes, raw_price, 1.0 - raw_price)
    asset_id = np.where(is_yes, token1_asset_id, token2_asset_id)

    # trade_sign = +1 iff is_yes == (side == 'BUY')
    is_buy = taker_direction == "BUY"
    trade_sign = np.where(is_yes == is_buy, 1, -1).astype(np.int8)

    return {
        "is_yes": is_yes,
        "asset_id": asset_id,
        "canonical_price": canonical_price,
        "trade_sign": trade_sign,
    }
