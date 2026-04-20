"""Historical import from the user's CSV scrape into ClickHouse.

Two phases:

  1. Markets. Reads markets.csv joined to pm_features.duckdb's market_meta
     (for category, which markets.csv lacks). One-shot insert into
     lattica.markets. ~300k rows, seconds.

  2. Trades. Streams processed/trades.csv in chunks via duckdb, joins
     markets.csv to get condition_id and asset_ids, applies vectorized
     canonicalization (common.canonical.canonicalize_batch), inserts into
     lattica.trades via clickhouse-connect. ~1B rows, hours.

No resume logic. If it crashes, TRUNCATE the target tables and re-run.

Usage:
    uv run python -m scripts.import_duckdb \\
        --trades      /path/to/poly_data/processed/trades.csv \\
        --markets     /path/to/poly_data/markets.csv \\
        --features-db /path/to/experiments/data/pm_features.duckdb \\
        [--ch-host localhost] [--ch-port 8123] \\
        [--skip-markets] [--skip-trades] \\
        [--batch-size 500000]
"""

from __future__ import annotations

import argparse
import sys
import time

import clickhouse_connect
import duckdb
import numpy as np
import pandas as pd

from common.canonical import canonicalize_batch


# ---------------------------------------------------------------------------
# Markets phase
# ---------------------------------------------------------------------------


def import_markets(
    ch: clickhouse_connect.driver.Client,
    markets_csv: str,
    features_db: str,
) -> int:
    """Import market metadata. Joins markets.csv to market_meta for category.

    markets.csv provides: createdAt, id, question, answer1/2, neg_risk,
    market_slug, token1, token2, condition_id, volume, closedTime.

    market_meta provides: category, tag_slugs, eligible, + duplicates.

    We take the join on market_id = markets.id to populate `category`. Markets
    without a market_meta row get category='unknown' (markets with no trade
    history — Phase 0 filtered them out; we keep them but tag them so the
    Gamma poller can later correct them on next live sync).
    """
    print("\n" + "=" * 72)
    print("Importing markets")
    print("=" * 72)

    con = duckdb.connect(":memory:")
    con.execute("SET threads TO 16")
    con.execute(f"ATTACH '{features_db}' AS fx (READ_ONLY)")

    # token1, token2 come back as DOUBLE from read_csv_auto because they're
    # huge decimal integers. Force them to VARCHAR to preserve precision.
    df = con.execute(f"""
        SELECT
            m.condition_id                                  AS condition_id,
            [m.token1, m.token2]                            AS asset_ids,
            m.question                                      AS question,
            coalesce(mm.category, 'unknown')                AS category,
            CAST(m.closedTime AS TIMESTAMP)                 AS resolution_time,
            CAST(m.createdAt  AS TIMESTAMP)                 AS created_at,
            coalesce(m.volume, 0.0)                         AS volume,
            coalesce(m.neg_risk, false)                     AS neg_risk,
            false                                           AS active,
            ''                                              AS end_date_iso,
            now()                                           AS updated_at
        FROM read_csv_auto(
            '{markets_csv}',
            types={{'token1': 'VARCHAR', 'token2': 'VARCHAR'}}
        ) m
        LEFT JOIN fx.market_meta mm ON mm.market_id = m.id
        WHERE m.condition_id IS NOT NULL
          AND m.token1 IS NOT NULL
          AND m.token2 IS NOT NULL
    """).df()

    con.close()
    print(f"  Read {len(df):,} markets from CSV (with category from market_meta)")

    # clickhouse-connect wants a list of lists + column order for typed inserts.
    cols = [
        "condition_id", "asset_ids", "question", "category",
        "resolution_time", "created_at", "volume", "neg_risk",
        "active", "end_date_iso", "updated_at",
    ]
    # Ensure dtype compatibility
    df["volume"] = df["volume"].astype(float)
    df["neg_risk"] = df["neg_risk"].astype(bool)

    t0 = time.time()
    ch.insert_df("lattica.markets", df[cols])
    elapsed = time.time() - t0
    print(f"  Inserted into lattica.markets in {elapsed:.1f}s")

    n = ch.query("SELECT count() FROM lattica.markets").result_rows[0][0]
    print(f"  lattica.markets now has {n:,} rows")
    return n


# ---------------------------------------------------------------------------
# Trades phase
# ---------------------------------------------------------------------------


def import_trades(
    ch: clickhouse_connect.driver.Client,
    trades_csv: str,
    markets_csv: str,
    batch_size: int,
) -> int:
    """Stream trades from CSV, canonicalize, batch-insert into ClickHouse.

    The source query filters out rows with invalid nonusdc_side, invalid
    taker_direction, and prices outside (0, 1). These filters match Phase 0's
    trades_enriched WHERE clause so our CH trades have the same universe.
    """
    print("\n" + "=" * 72)
    print("Importing trades")
    print("=" * 72)

    con = duckdb.connect(":memory:")
    con.execute("SET threads TO 16")
    con.execute("SET preserve_insertion_order = false")

    # Build the streaming source. Join markets inline so every row has
    # condition_id + both token ids ready for canonicalization.
    src_query = f"""
        SELECT
            m.condition_id                 AS condition_id,
            m.token1                       AS token1,
            m.token2                       AS token2,
            t.nonusdc_side                 AS nonusdc_side,
            t.price                        AS price,
            t.usd_amount                   AS dollar_volume,
            t.token_amount                 AS size,
            t.taker_direction              AS side,
            CAST(t.timestamp AS TIMESTAMP) AS timestamp,
            t.transactionHash              AS transaction_hash
        FROM read_csv_auto('{trades_csv}') t
        JOIN read_csv_auto(
            '{markets_csv}',
            types={{'token1': 'VARCHAR', 'token2': 'VARCHAR'}}
        ) m ON t.market_id = m.id
        WHERE t.nonusdc_side IN ('token1', 'token2')
          AND t.taker_direction IN ('BUY', 'SELL')
          AND t.price > 0
          AND t.price < 1
    """

    cursor = con.execute(src_query)

    t0 = time.time()
    total = 0
    batch_n = 0

    while True:
        chunk = cursor.fetch_df_chunk(batch_size // 2048)
        if chunk is None or len(chunk) == 0:
            break

        df = _build_trades_batch(chunk)
        ch.insert_df("lattica.trades", df)

        total += len(df)
        batch_n += 1
        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(
            f"  batch {batch_n:>4}  "
            f"+{len(df):>7,}  total={total:>12,}  "
            f"rate={rate:>8,.0f} rows/s  elapsed={elapsed:>7.0f}s"
        )

    con.close()

    elapsed = time.time() - t0
    print(f"\n  Done. {total:,} rows in {elapsed/60:.1f} min "
          f"({total/elapsed:.0f} rows/s avg)")

    n = ch.query("SELECT count() FROM lattica.trades").result_rows[0][0]
    print(f"  lattica.trades now has {n:,} rows")
    return n


def _build_trades_batch(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply canonicalization and shape the batch to match the CH schema."""
    c = canonicalize_batch(
        nonusdc_side=chunk["nonusdc_side"].to_numpy(),
        raw_price=chunk["price"].to_numpy(dtype=np.float64),
        taker_direction=chunk["side"].to_numpy(),
        token1_asset_id=chunk["token1"].to_numpy(),
        token2_asset_id=chunk["token2"].to_numpy(),
    )

    return pd.DataFrame({
        "condition_id":     chunk["condition_id"].to_numpy(),
        "asset_id":         c["asset_id"],
        "is_yes":           c["is_yes"],
        "price":            chunk["price"].to_numpy(dtype=np.float64),
        "canonical_price":  c["canonical_price"],
        "size":             chunk["size"].to_numpy(dtype=np.float64),
        "dollar_volume":    chunk["dollar_volume"].to_numpy(dtype=np.float64),
        "side":             chunk["side"].to_numpy(),
        "trade_sign":       c["trade_sign"],
        "timestamp":        chunk["timestamp"].to_numpy(),
        "transaction_hash": chunk["transaction_hash"].to_numpy(),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trades", required=True, help="processed/trades.csv")
    p.add_argument("--markets", required=True, help="markets.csv")
    p.add_argument("--features-db", required=True,
                   help="pm_features.duckdb (for category metadata)")
    p.add_argument("--ch-host", default="localhost")
    p.add_argument("--ch-port", type=int, default=8123)
    p.add_argument("--ch-user", default="default")
    p.add_argument("--ch-password", default="")
    p.add_argument("--batch-size", type=int, default=500_000)
    p.add_argument("--skip-markets", action="store_true")
    p.add_argument("--skip-trades", action="store_true")
    args = p.parse_args()

    ch = clickhouse_connect.get_client(
        host=args.ch_host,
        port=args.ch_port,
        username=args.ch_user,
        password=args.ch_password,
        database="lattica",
    )

    # Quick sanity check: schema exists.
    tables = {r[0] for r in ch.query(
        "SELECT name FROM system.tables WHERE database='lattica'"
    ).result_rows}
    required = {"markets", "trades", "orderbook_snapshots"}
    missing = required - tables
    if missing:
        print(f"ERROR: lattica schema missing tables: {missing}", file=sys.stderr)
        print("  Did you run `docker-compose up clickhouse`?", file=sys.stderr)
        return 1

    if not args.skip_markets:
        import_markets(ch, args.markets, args.features_db)

    if not args.skip_trades:
        import_trades(ch, args.trades, args.markets, args.batch_size)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
