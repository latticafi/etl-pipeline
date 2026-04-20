"""Import historical orderbook snapshots from the user's .zst scrape.

Directory layout:
    {root}/{market_id}.jsonl.zst
      each line is one full book snapshot JSON:
        market        — condition_id (hex)
        asset_id      — the token traded (str digits)
        timestamp     — unix millis (str)
        bids, asks    — list of {price, size}, unsorted
        tick_size, neg_risk, last_trade_price, hash, min_order_size

We derive: best_bid, best_ask, mid_price, spread_bps, depth_bid_5pct,
depth_ask_5pct, imbalance, bid_levels, ask_levels. is_yes comes from
joining asset_id against lattica.markets.asset_ids[0]/[1].

Parallelism: ProcessPoolExecutor over files. Each worker decompresses and
parses one file fully, returns all rows as a dict of arrays. Main process
concatenates and batch-inserts to CH.

Usage:
    uv run python -m scripts.import_orderbook \\
        --root ~/repos/poly_data/orderbook_snapshots/data \\
        [--ch-host localhost] [--workers 16]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import clickhouse_connect
import pandas as pd
import zstandard


# ---------------------------------------------------------------------------
# Per-snapshot parser
# ---------------------------------------------------------------------------


def _derive_aggregates(bids: list, asks: list, depth_pct: float = 0.05) -> dict | None:
    """Compute best_bid, best_ask, mid, spread_bps, depth, imbalance.

    Levels come in unsorted. Take max(bid_prices) and min(ask_prices).
    Skip snapshots with empty or crossed books.
    """
    if not bids or not asks:
        return None

    try:
        bid_prices = [float(b["price"]) for b in bids]
        bid_sizes = [float(b["size"]) for b in bids]
        ask_prices = [float(a["price"]) for a in asks]
        ask_sizes = [float(a["size"]) for a in asks]
    except (KeyError, TypeError, ValueError):
        return None

    best_bid = max(bid_prices)
    best_ask = min(ask_prices)
    if best_bid <= 0 or best_ask <= 0 or best_ask <= best_bid:
        return None

    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 10_000

    bid_bound = mid * (1 - depth_pct)
    ask_bound = mid * (1 + depth_pct)
    depth_bid = sum(p * s for p, s in zip(bid_prices, bid_sizes) if p >= bid_bound)
    depth_ask = sum(p * s for p, s in zip(ask_prices, ask_sizes) if p <= ask_bound)
    total = depth_bid + depth_ask
    imbalance = depth_bid / total if total > 0 else 0.5

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread_bps": spread_bps,
        "depth_bid_5pct": depth_bid,
        "depth_ask_5pct": depth_ask,
        "imbalance": imbalance,
        "bid_levels": len(bids),
        "ask_levels": len(asks),
    }


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------


def _process_file(args: tuple[str, dict[str, tuple[str, str]]]) -> tuple[str, list[dict], int]:
    """Decompress and parse one .zst file. Runs in a worker process.

    Returns (path, parsed_rows, skipped_count). is_yes is resolved here
    using the shared market_index (condition_id -> (yes_token, no_token)).
    """
    path, market_index = args
    rows: list[dict] = []
    skipped = 0

    try:
        with open(path, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    cid = d.get("market")
                    asset_id = d.get("asset_id")
                    ts_ms = d.get("timestamp")
                    if not (cid and asset_id and ts_ms):
                        skipped += 1
                        continue

                    tokens = market_index.get(cid)
                    if tokens is None:
                        skipped += 1
                        continue
                    is_yes = asset_id == tokens[0]

                    agg = _derive_aggregates(d.get("bids", []), d.get("asks", []))
                    if agg is None:
                        skipped += 1
                        continue

                    try:
                        ts = datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc)
                    except (TypeError, ValueError):
                        skipped += 1
                        continue

                    rows.append({
                        "condition_id": cid,
                        "asset_id": asset_id,
                        "is_yes": is_yes,
                        "timestamp": ts,
                        **agg,
                    })
    except Exception as e:
        # Bad file — log and skip, don't kill the whole import.
        return (path, [], -1)  # -1 signals "file failed"

    return (path, rows, skipped)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


COLUMNS = [
    "condition_id", "asset_id", "is_yes",
    "best_bid", "best_ask", "mid_price", "spread_bps",
    "depth_bid_5pct", "depth_ask_5pct", "imbalance",
    "bid_levels", "ask_levels",
    "timestamp",
]


def _load_market_index(ch: clickhouse_connect.driver.Client) -> dict[str, tuple[str, str]]:
    """condition_id -> (yes_token, no_token) from lattica.markets_latest."""
    rows = ch.query(
        "SELECT condition_id, asset_ids FROM lattica.markets_latest"
    ).result_rows
    return {r[0]: (r[1][0], r[1][1]) for r in rows if len(r[1]) >= 2}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="orderbook_snapshots/data directory")
    p.add_argument("--ch-host", default="localhost")
    p.add_argument("--ch-port", type=int, default=8123)
    p.add_argument("--ch-user", default="default")
    p.add_argument("--ch-password", default="")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--insert-batch", type=int, default=200_000,
                   help="accumulate this many rows before each CH insert")
    p.add_argument("--limit-files", type=int, default=0,
                   help="process at most N files (0 = all); useful for dry-run")
    args = p.parse_args()

    ch = clickhouse_connect.get_client(
        host=args.ch_host, port=args.ch_port,
        username=args.ch_user, password=args.ch_password,
        database="lattica",
    )

    # Market index must be loaded before the import (imports markets first,
    # then orderbook). If it's empty, we'd silently skip everything.
    print("Loading market index from lattica.markets...")
    market_index = _load_market_index(ch)
    print(f"  {len(market_index):,} markets indexed")
    if not market_index:
        print("ERROR: no markets in lattica.markets. Run import_duckdb first.",
              file=sys.stderr)
        return 1

    root = Path(args.root)
    files = sorted(root.glob("*.jsonl.zst"))
    if args.limit_files > 0:
        files = files[:args.limit_files]
    print(f"{len(files):,} files to process, {args.workers} workers")

    t0 = time.time()
    buffer: list[dict] = []
    total_rows = 0
    total_skipped = 0
    failed_files = 0
    files_done = 0

    def flush():
        nonlocal buffer, total_rows
        if not buffer:
            return
        df = pd.DataFrame(buffer, columns=COLUMNS)
        ch.insert_df("lattica.orderbook_snapshots", df)
        total_rows += len(buffer)
        buffer = []

    # ProcessPoolExecutor serializes args per task — we want the market_index
    # shared, but Python multiprocessing copies it per worker. 300k entries
    # is ~tens of MB, tolerable to copy per worker.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_process_file, (str(p), market_index)): p.name
            for p in files
        }
        for fut in as_completed(futs):
            path, rows, skipped = fut.result()
            files_done += 1

            if skipped < 0:
                failed_files += 1
            else:
                total_skipped += skipped
                buffer.extend(rows)

            if len(buffer) >= args.insert_batch:
                flush()

            if files_done % 100 == 0 or files_done == len(files):
                elapsed = time.time() - t0
                rate = total_rows / elapsed if elapsed > 0 else 0
                print(
                    f"  files {files_done:>6,}/{len(files):,}  "
                    f"rows={total_rows:>12,}  "
                    f"skipped={total_skipped:>9,}  "
                    f"failed_files={failed_files}  "
                    f"rate={rate:>7,.0f} rows/s  "
                    f"elapsed={elapsed:>6.0f}s"
                )

    flush()

    elapsed = time.time() - t0
    n = ch.query("SELECT count() FROM lattica.orderbook_snapshots").result_rows[0][0]
    print(f"\nDone in {elapsed/60:.1f} min.")
    print(f"  lattica.orderbook_snapshots has {n:,} rows")
    print(f"  {total_skipped:,} lines skipped (empty/invalid), "
          f"{failed_files} files failed to parse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
