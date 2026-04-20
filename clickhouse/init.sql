-- Lattica data pipeline — ClickHouse initialization.
--
-- Runs on first container boot via /docker-entrypoint-initdb.d/.
-- Idempotent: all statements use IF NOT EXISTS so restarts are safe.
--
-- See plan.md §4 for design rationale.

CREATE DATABASE IF NOT EXISTS lattica;

-- ---------------------------------------------------------------------------
-- trades
-- ---------------------------------------------------------------------------
-- Source of truth for every fill on every market, both YES and NO tokens.
-- Canonicalization is done once at ingest time by the writer (WSS handler
-- or duckdb import script). Downstream code reads canonical_price and
-- trade_sign directly; it does not recompute.
--
-- ~40M rows/day at current Polymarket volume. LZ4 gives ~10x compression.
-- 1-year TTL; older data is archived to S3 as Parquet and dropped here.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lattica.trades (
    condition_id      LowCardinality(String),
    asset_id          String,
    is_yes            Bool,                          -- asset_id == markets.asset_ids[0]
    price             Float64,                       -- raw price on asset_id
    canonical_price   Float64,                       -- 1-price if NO, else price
    size              Float64,
    dollar_volume     Float64,                       -- price * size (raw frame)
    side              Enum8('BUY' = 1, 'SELL' = 2),  -- raw taker direction
    trade_sign        Int8,                          -- +1 / -1 in canonical frame
    timestamp         DateTime64(3, 'UTC'),
    transaction_hash  String,
    ingested_at       DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (condition_id, timestamp)
TTL toDateTime(timestamp) + INTERVAL 3 DAY   TO VOLUME 'cold',
    toDateTime(timestamp) + INTERVAL 1 YEAR  DELETE
SETTINGS
    storage_policy = 'tiered',
    index_granularity = 8192;

-- ---------------------------------------------------------------------------
-- orderbook_snapshots
-- ---------------------------------------------------------------------------
-- ~30s-per-market sampling cadence from WSS `book` events. Captured for
-- both tokens. Training v002 will ASOF-join these against epoch_start.
-- 6-month TTL — orderbook history loses value fast.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lattica.orderbook_snapshots (
    condition_id      LowCardinality(String),
    asset_id          String,
    is_yes            Bool,
    best_bid          Float64,
    best_ask          Float64,
    mid_price         Float64,
    spread_bps        Float64,
    depth_bid_5pct    Float64,
    depth_ask_5pct    Float64,
    imbalance         Float64,
    bid_levels        UInt32,
    ask_levels        UInt32,
    timestamp         DateTime64(3, 'UTC'),
    ingested_at       DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (condition_id, timestamp)
TTL toDateTime(timestamp) + INTERVAL 3 DAY   TO VOLUME 'cold',
    toDateTime(timestamp) + INTERVAL 6 MONTH DELETE
SETTINGS
    storage_policy = 'tiered',
    index_granularity = 8192;

-- ---------------------------------------------------------------------------
-- markets
-- ---------------------------------------------------------------------------
-- Market metadata from Gamma. ReplacingMergeTree dedupes on ORDER BY key
-- at merge time, keeping the row with the max updated_at. Production
-- queries should use the `markets_latest` view below.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS lattica.markets (
    condition_id      String,
    asset_ids         Array(String),              -- [yes_token, no_token]
    question          String,
    category          LowCardinality(String),
    resolution_time   DateTime64(3, 'UTC'),
    created_at        DateTime64(3, 'UTC'),
    volume            Float64,
    neg_risk          Bool,
    active            Bool,
    end_date_iso      String,
    updated_at        DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(updated_at)
ORDER BY condition_id
SETTINGS index_granularity = 8192;

-- Convenience view: always returns one row per market, latest values.
-- Readers should SELECT FROM markets_latest, never from markets directly.
CREATE VIEW IF NOT EXISTS lattica.markets_latest AS
SELECT
    condition_id,
    argMax(asset_ids, updated_at)       AS asset_ids,
    argMax(question, updated_at)        AS question,
    argMax(category, updated_at)        AS category,
    argMax(resolution_time, updated_at) AS resolution_time,
    argMax(created_at, updated_at)      AS created_at,
    argMax(volume, updated_at)          AS volume,
    argMax(neg_risk, updated_at)        AS neg_risk,
    argMax(active, updated_at)          AS active,
    argMax(end_date_iso, updated_at)    AS end_date_iso,
    max(updated_at)                     AS updated_at
FROM lattica.markets
GROUP BY condition_id;
