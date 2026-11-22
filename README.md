# bid-screens

A compute family of the s1d-labs scrape pool: statistical **bid screens** (indicators used by competition authorities) on
public bid data from GeM and the GePNIC tender portals. It scrapes nothing: inputs are Parquet shards that the database box
exports into the family owner's bucket, the pool's GitHub runners compute the screens, and results go back to the bucket.
A screen flag is an **indicator** ("screen flags"), never a finding about any firm.

| Path | What |
|---|---|
| `screens/bidscreens_v1.py` | version 1 screens, frozen before the pilot (byte-identical to the pre-registered file) |
| `screens/v2.py` | version 2: B-to-reverse-auction join, sync2, technical disqualification, cover with firm control, closed bidder set |
| `screens/labels.py` | held-out validation cases (fixed before any v2 value was computed) |
| `screens/fastpairs.py` | vectorised pair builder (same rows as v1, speed only) |
| `run_shard.py` | task `screens`: one shard at a time -> `out/<shard>/*.parquet` + a summary record |
| `rank.py` | task `rank`: national composites, base rates, validation, lead lists -> `national/<run>/` |
| `export/` | box side: `gem_export.py`, `gepnic_export.py` (+ `.sql`), `push.sh`/`push.py` (upload with the hub's client), `seed.py` |

Projects: `bid-screens-gem`, `bid-screens-gepnic` (task `screens`), `bid-screens-rank` (task `rank`); 20 + 58 + 1 worker
processes at most (<= 40 runner jobs). Queue items name bucket keys and the pinned code hash; the task refuses another hash.

Check: `python -m pytest tests` and
`scrape-check --project bid-screens-gem --task screens --env BID_SCREENS_MIN_BUDGET_MIN=2 --item '{"key":"s1","payload":{"shard":"check/s1","source":"gem","synthetic":150}}'`.
