"""What this family does: it computes bid screens (statistical indicators) on public bid data. No scraping, no target site.

Each project is one queue of shards; a shard's inputs are Parquet files in the family owner's bucket under the project's
prefix, written on the box that holds the database (export/), and its queue items are seeded from there (they name the
bucket keys and the pinned code hash). Pre-registration: bet-3 research folder, preregistration-v2.md.
"""
import json
from pathlib import Path

FAMILY = json.loads((Path(__file__).parent / "scrape.json").read_text())["family"]

# 20 + 58 + 1 worker processes = at most 40 runner jobs at 2 processes per job (owner: a share of <= 40 jobs)
PROJECTS = [
    {"name": f"{FAMILY}-gem", "title": "Bid screens v1 + v2 on GeM bids (shards of item classes)", "task": "screens",
     "unit_label": "shard", "record_label": "shard summary", "max_workers": 20},
    {"name": f"{FAMILY}-gepnic", "title": "Bid screens v1 + v2 on GePNIC bid lists (portal-year shards)", "task": "screens",
     "unit_label": "shard", "record_label": "shard summary", "max_workers": 58},
    {"name": f"{FAMILY}-rank", "title": "Bid screens: national ranks, base rates and validation", "task": "rank",
     "unit_label": "run", "record_label": "national run", "max_workers": 1},
]
