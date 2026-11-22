#!/usr/bin/env python3
"""Export GeM bids from gem_lab (lab.bid_core, lab.bidder_core, lab.bid, dims, hr_bidder makes) into Parquet shards.

Runs on the database server at nice 15, read-only, keyed by year (bid_core.yr) with a statement timeout. A shard holds whole
item classes over all years (2018-2026); a class with more than CLASS_SPLIT bid rows is cut by year, and then each year shard
also holds the reverse auctions published up to 31 March of the next year (so a B bid and its RA meet in one shard).
Output: STAGE/gem/<shard>/{tenders.parquet, bids-<yr>.parquet}, STAGE/gem/directors.parquet, STAGE/manifest-gem.jsonl.
  nice -n 15 .venv/bin/python export/gem_export.py STAGE
"""
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SHARD_ROWS = 200_000         # weight per shard: bid rows + 5 x tenders (a shard of whole classes)
CLASS_SPLIT = 200_000        # a class with more weight than this is cut by year
YEARS = range(2018, 2027)
HEAD = "\\set ON_ERROR_STOP on\nset default_transaction_read_only = on;\nset statement_timeout = '20min';\nset work_mem = '256MB';\n"


def psql_csv(sql, out=None):
    cmd = ["nice", "-n", "15", "sudo", "-n", "-u", "postgres", "psql", "-q", "-X", "-d", "gem_lab"]
    script = (HEAD + f"copy ({sql}) to stdout with csv header;\n").encode()
    if out:
        with open(out, "wb") as f:
            subprocess.run(cmd, input=script, stdout=f, check=True)
        return out
    return io.BytesIO(subprocess.run(cmd, input=script, capture_output=True, check=True).stdout)


TENDERS = """
select c.tender_id, c.yr, b.published_date as published, b.closing_date as closing, c.ref_type, c.status, c.item_id, i.item_key,
       c.buyer_id, d.ministry, d.organisation, d.state, c.n_received, c.n_tech, c.n_priced, c.n_disq, c.award_value, c.single_item,
       c.quantity, b.ref
from lab.bid_core c join lab.bid b using (tender_id) join lab.dim_item i using (item_id) left join lab.dim_buyer d using (buyer_id)
where c.yr between 2018 and 2026 and c.tender_id not in (select tender_id from lab.test_bid)"""

BIDS = """
with mk as (select h.pid, h.seller_norm, min(nullif(lower(regexp_replace(h.make, '[^A-Za-z0-9]', '', 'g')), '')) as make
            from lab.hr_bidder h join lab.bid b on b.portal_internal_id = h.pid where b.yr = {y} group by 1, 2)
select x.tender_id, x.seller_id, ds.seller_key, x.tech, x.submitted_at, x.price, x.rank_n, x.mse, mk.make
from lab.bidder_core x join lab.bid_core c using (tender_id) join lab.bid b using (tender_id) join lab.dim_seller ds using (seller_id)
left join mk on mk.pid = b.portal_internal_id and mk.seller_norm = ds.seller_key
where c.yr = {y} and c.tender_id not in (select tender_id from lab.test_bid)"""

DIRECTORS = """
select distinct m.seller_key, m.cin, p.din from lab.seller_mca m join lab.pool_director p on p.cin = m.cin
where m.confident and p.din is not null and p.din <> ''"""

ROWS = """
select c.item_id, c.yr, count(x.*) + 5 * count(distinct c.tender_id) as n from lab.bid_core c left join lab.bidder_core x using (tender_id)
where c.yr between 2018 and 2026 and c.tender_id not in (select tender_id from lab.test_bid) group by 1, 2"""


def plan(rows):
    """item_id -> shard for classes kept whole; (item_id, yr) -> shard for split classes."""
    tot = rows.groupby("item_id")["n"].sum()
    whole = tot[tot <= CLASS_SPLIT].sort_values(ascending=False)
    split = tot[tot > CLASS_SPLIT].index
    item_shard, loads = {}, []
    for item, n in whole.items():                         # first fit decreasing
        k = next((i for i, l in enumerate(loads) if l + n <= SHARD_ROWS), None)
        if k is None:
            loads.append(0); k = len(loads) - 1
        loads[k] += n; item_shard[item] = f"c{k:04d}"
    year_shard = {}
    for item in sorted(split):
        for yr in rows.loc[rows["item_id"] == item, "yr"]:
            year_shard[(item, int(yr))] = f"y{item}-{int(yr)}"
    return item_shard, year_shard


def tender_shards(t, item_shard, year_shard):
    """(tender_id, shard) pairs. A reverse auction of a split class goes to the shard of its year and, when published before
    1 April, also to the previous year's shard."""
    t = t[["tender_id", "item_id", "yr", "ref_type", "published"]].copy()
    t["shard"] = t["item_id"].map(item_shard)
    s = t[t["shard"].isna()].copy()
    s["shard"] = [year_shard.get((i, int(y))) for i, y in zip(s["item_id"], s["yr"])]
    ra = s[(s["ref_type"] != "B") & (pd.to_datetime(s["published"]).dt.month <= 3)].copy()
    ra["shard"] = [year_shard.get((i, int(y) - 1)) for i, y in zip(ra["item_id"], ra["yr"])]
    out = pd.concat([t[t["shard"].notna()], s, ra], ignore_index=True)
    return out.loc[out["shard"].notna(), ["tender_id", "shard"]].drop_duplicates()


def main():
    stage = Path(sys.argv[1]) / "gem"
    stage.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = pd.read_csv(psql_csv(ROWS))
    item_shard, year_shard = plan(rows)
    print(f"plan: {len(set(item_shard.values()))} class shards, {len(year_shard)} year shards ({time.time() - t0:.0f}s)", flush=True)
    tcsv = stage / ".tenders.csv"
    psql_csv(TENDERS, tcsv)
    t = pd.read_csv(tcsv, low_memory=False, dtype={"ref_type": str, "status": str, "item_key": str, "ministry": str,
                                                   "organisation": str, "state": str, "ref": str})
    for c in ("published", "closing"):
        t[c] = pd.to_datetime(t[c], errors="coerce")
    ts = tender_shards(t, item_shard, year_shard)
    print(f"tenders {len(t)}, tender-shard rows {len(ts)} ({time.time() - t0:.0f}s)", flush=True)
    tx = t.merge(ts, on="tender_id")
    for shard, g in tx.groupby("shard"):
        d = stage / shard
        d.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g.drop(columns=["shard"]), preserve_index=False), d / "tenders.parquet", compression="zstd")
    tcsv.unlink()
    ntb = {}
    for y in YEARS:
        bcsv = stage / f".bids-{y}.csv"
        psql_csv(BIDS.format(y=y), bcsv)
        b = pd.read_csv(bcsv, low_memory=False, dtype={"seller_key": str, "make": str})
        b["submitted_at"] = pd.to_datetime(b["submitted_at"], errors="coerce")
        b = b.merge(ts, on="tender_id")
        for shard, g in b.groupby("shard"):
            pq.write_table(pa.Table.from_pandas(g.drop(columns=["shard"]), preserve_index=False), stage / shard / f"bids-{y}.parquet",
                           compression="zstd")
            ntb[shard] = ntb.get(shard, 0) + len(g)
        bcsv.unlink()
        print(f"bids {y}: {len(b)} rows ({time.time() - t0:.0f}s)", flush=True)
    dcsv = stage / ".directors.csv"
    psql_csv(DIRECTORS, dcsv)
    pq.write_table(pa.Table.from_pandas(pd.read_csv(dcsv, dtype=str), preserve_index=False), stage / "directors.parquet", compression="zstd")
    dcsv.unlink()
    with open(stage.parent / "manifest-gem.jsonl", "w") as f:
        for shard in sorted(tx["shard"].unique()):
            d = stage / shard
            files = {p.name: p.stat().st_size for p in sorted(d.glob("*.parquet"))}
            f.write(json.dumps({"shard": f"gem/{shard}", "source": "gem", "n_tenders": int((tx["shard"] == shard).sum()),
                                "n_bids": int(ntb.get(shard, 0)), "files": files}) + "\n")
    print(f"DONE {tx['shard'].nunique()} shards in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
