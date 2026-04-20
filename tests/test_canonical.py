"""Canonicalization tests.

These tests pin down the sign conventions and price flips that the current
production worker gets wrong. If any of them fail, downstream features
are computed in the wrong frame.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from common.canonical import (
    CanonicalTrade,
    canonical_price_from,
    canonicalize_duckdb,
    canonicalize_wss,
    trade_sign_from,
)

YES_TOKEN = "0xAAAA"
NO_TOKEN = "0xBBBB"
COND_ID = "0xCOND"
TS = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
ASSET_IDS = (YES_TOKEN, NO_TOKEN)


# ---------------------------------------------------------------------------
# canonical_price_from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_price", "is_yes", "expected"),
    [
        (0.30, True, 0.30),   # YES passthrough
        (0.30, False, 0.70),  # NO flipped to YES frame
        (0.50, True, 0.50),   # midpoint symmetric
        (0.50, False, 0.50),
        (0.99, True, 0.99),
        (0.99, False, 0.01),
        (0.01, True, 0.01),
        (0.01, False, 0.99),
    ],
)
def test_canonical_price(raw_price: float, is_yes: bool, expected: float) -> None:
    assert canonical_price_from(raw_price, is_yes) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# trade_sign_from — the truth table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("is_yes", "side", "expected_sign"),
    [
        # YES token: BUY = buying YES = +1, SELL = selling YES = -1
        (True, "BUY", +1),
        (True, "SELL", -1),
        # NO token: BUY NO = selling YES = -1, SELL NO = buying YES = +1
        (False, "BUY", -1),
        (False, "SELL", +1),
    ],
)
def test_trade_sign_truth_table(is_yes: bool, side: str, expected_sign: int) -> None:
    assert trade_sign_from(is_yes, side) == expected_sign


@pytest.mark.parametrize("side", ["buy", "Buy", "  BUY  ", "sell", "SELL"])
def test_trade_sign_accepts_case_and_whitespace(side: str) -> None:
    # Shouldn't raise
    trade_sign_from(True, side)


@pytest.mark.parametrize("bad_side", ["", "bid", "LONG", "0", "1"])
def test_trade_sign_rejects_invalid(bad_side: str) -> None:
    with pytest.raises(ValueError):
        trade_sign_from(True, bad_side)


# ---------------------------------------------------------------------------
# canonicalize_wss
# ---------------------------------------------------------------------------


def test_wss_yes_buy() -> None:
    t = canonicalize_wss(
        condition_id=COND_ID,
        asset_id=YES_TOKEN,
        asset_ids=ASSET_IDS,
        raw_price=0.30,
        size=100.0,
        side="BUY",
        timestamp=TS,
        transaction_hash="0xtx",
    )
    assert t.is_yes is True
    assert t.canonical_price == pytest.approx(0.30)
    assert t.trade_sign == +1
    assert t.dollar_volume == pytest.approx(30.0)  # price * size, raw frame


def test_wss_no_buy_flips_to_yes_sell() -> None:
    t = canonicalize_wss(
        condition_id=COND_ID,
        asset_id=NO_TOKEN,
        asset_ids=ASSET_IDS,
        raw_price=0.30,     # NO at 0.30 means YES at 0.70
        size=100.0,
        side="BUY",
        timestamp=TS,
        transaction_hash="0xtx",
    )
    assert t.is_yes is False
    assert t.canonical_price == pytest.approx(0.70)
    assert t.trade_sign == -1    # BUY NO == SELL YES
    assert t.dollar_volume == pytest.approx(30.0)  # raw, not canonical


def test_wss_no_sell_flips_to_yes_buy() -> None:
    t = canonicalize_wss(
        condition_id=COND_ID,
        asset_id=NO_TOKEN,
        asset_ids=ASSET_IDS,
        raw_price=0.30,
        size=50.0,
        side="SELL",
        timestamp=TS,
        transaction_hash="0xtx",
    )
    assert t.is_yes is False
    assert t.canonical_price == pytest.approx(0.70)
    assert t.trade_sign == +1    # SELL NO == BUY YES


def test_wss_unknown_asset_id_raises() -> None:
    with pytest.raises(ValueError, match="matches neither YES"):
        canonicalize_wss(
            condition_id=COND_ID,
            asset_id="0xDEAD",
            asset_ids=ASSET_IDS,
            raw_price=0.50,
            size=1.0,
            side="BUY",
            timestamp=TS,
            transaction_hash="0xtx",
        )


# ---------------------------------------------------------------------------
# canonicalize_duckdb
# ---------------------------------------------------------------------------


def test_duckdb_token1_maps_to_yes() -> None:
    t = canonicalize_duckdb(
        condition_id=COND_ID,
        asset_ids=ASSET_IDS,
        nonusdc_side="token1",
        raw_price=0.25,
        usd_amount=25.0,
        token_amount=100.0,
        taker_direction="BUY",
        timestamp=TS,
        tx_hash="0xtx",
    )
    assert t.is_yes is True
    assert t.asset_id == YES_TOKEN
    assert t.canonical_price == pytest.approx(0.25)
    assert t.trade_sign == +1


def test_duckdb_token2_maps_to_no_and_flips() -> None:
    t = canonicalize_duckdb(
        condition_id=COND_ID,
        asset_ids=ASSET_IDS,
        nonusdc_side="token2",
        raw_price=0.25,           # NO at 0.25 → YES at 0.75
        usd_amount=25.0,
        token_amount=100.0,
        taker_direction="BUY",    # BUY NO == SELL YES
        timestamp=TS,
        tx_hash="0xtx",
    )
    assert t.is_yes is False
    assert t.asset_id == NO_TOKEN
    assert t.canonical_price == pytest.approx(0.75)
    assert t.trade_sign == -1


def test_duckdb_usd_amount_not_recomputed() -> None:
    # duckdb's usd_amount is trusted as-is; we do NOT recompute from price*size.
    # Simulates a scrape row where usd_amount was rounded slightly.
    t = canonicalize_duckdb(
        condition_id=COND_ID,
        asset_ids=ASSET_IDS,
        nonusdc_side="token1",
        raw_price=0.25,
        usd_amount=24.99,      # not 25.0
        token_amount=100.0,
        taker_direction="BUY",
        timestamp=TS,
        tx_hash="0xtx",
    )
    assert t.dollar_volume == pytest.approx(24.99)


def test_duckdb_invalid_side_label() -> None:
    with pytest.raises(ValueError, match="nonusdc_side"):
        canonicalize_duckdb(
            condition_id=COND_ID,
            asset_ids=ASSET_IDS,
            nonusdc_side="usdc",
            raw_price=0.5,
            usd_amount=1.0,
            token_amount=2.0,
            taker_direction="BUY",
            timestamp=TS,
            tx_hash="0xtx",
        )


# ---------------------------------------------------------------------------
# Round-trip: same economic trade via both paths produces identical canonical
# ---------------------------------------------------------------------------


def test_roundtrip_wss_matches_duckdb() -> None:
    """Same economic trade expressed as WSS event and as duckdb row should
    produce identical CanonicalTrade values (modulo fields that genuinely
    differ — `size` vs `token_amount` have the same meaning)."""
    wss = canonicalize_wss(
        condition_id=COND_ID,
        asset_id=NO_TOKEN,
        asset_ids=ASSET_IDS,
        raw_price=0.40,
        size=50.0,
        side="SELL",
        timestamp=TS,
        transaction_hash="0xtx",
    )
    ddb = canonicalize_duckdb(
        condition_id=COND_ID,
        asset_ids=ASSET_IDS,
        nonusdc_side="token2",
        raw_price=0.40,
        usd_amount=20.0,    # 0.40 * 50
        token_amount=50.0,
        taker_direction="SELL",
        timestamp=TS,
        tx_hash="0xtx",
    )
    assert wss.is_yes == ddb.is_yes
    assert wss.canonical_price == pytest.approx(ddb.canonical_price)
    assert wss.trade_sign == ddb.trade_sign
    assert wss.asset_id == ddb.asset_id
    assert wss.dollar_volume == pytest.approx(ddb.dollar_volume)


# ---------------------------------------------------------------------------
# CanonicalTrade is immutable
# ---------------------------------------------------------------------------


def test_canonical_trade_is_frozen() -> None:
    t = canonicalize_wss(
        condition_id=COND_ID,
        asset_id=YES_TOKEN,
        asset_ids=ASSET_IDS,
        raw_price=0.5,
        size=1.0,
        side="BUY",
        timestamp=TS,
        transaction_hash="0xtx",
    )
    with pytest.raises((AttributeError, Exception)):
        t.canonical_price = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Batch vs scalar equivalence
# ---------------------------------------------------------------------------


def test_canonicalize_batch_matches_scalar() -> None:
    """canonicalize_batch over arrays must produce the same output per-row
    as canonicalize_duckdb called in a loop. This pins the invariant that
    the bulk-import and WSS-live code paths compute identically."""
    import numpy as np

    from common.canonical import canonicalize_batch

    # 100 rows covering all four (nonusdc_side, taker_direction) combos
    n = 100
    rng = np.random.default_rng(42)
    sides = rng.choice(["token1", "token2"], size=n)
    dirs = rng.choice(["BUY", "SELL"], size=n)
    prices = rng.uniform(0.01, 0.99, size=n)

    # Scalar loop
    scalar_is_yes = []
    scalar_asset = []
    scalar_cp = []
    scalar_sign = []
    for i in range(n):
        t = canonicalize_duckdb(
            condition_id=COND_ID,
            asset_ids=ASSET_IDS,
            nonusdc_side=str(sides[i]),
            raw_price=float(prices[i]),
            usd_amount=1.0,
            token_amount=1.0,
            taker_direction=str(dirs[i]),
            timestamp=TS,
            tx_hash="0xtx",
        )
        scalar_is_yes.append(t.is_yes)
        scalar_asset.append(t.asset_id)
        scalar_cp.append(t.canonical_price)
        scalar_sign.append(t.trade_sign)

    # Vectorized
    batch = canonicalize_batch(
        nonusdc_side=sides,
        raw_price=prices,
        taker_direction=dirs,
        token1_asset_id=np.array([YES_TOKEN] * n),
        token2_asset_id=np.array([NO_TOKEN] * n),
    )

    assert list(batch["is_yes"]) == scalar_is_yes
    assert list(batch["asset_id"]) == scalar_asset
    assert np.allclose(batch["canonical_price"], scalar_cp)
    assert list(batch["trade_sign"]) == scalar_sign
