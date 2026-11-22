#!/usr/bin/env python3
"""Export GePNIC bid lists from scrape_hub into Parquet shards (runs on the database server, not on runners).

One portal-year at a time (pkey range scan of records, export/gepnic_export.sql), only at quiet hours (QUIET, IST), at nice 15.
A portal-year is cut into shards of whole org_ids (an item class is org_id x category), each <= SHARD_ROWS pair-relevant bid rows.
Output: STAGE/gepnic/<portal>/<year>-<k>/{tenders,bids}.parquet and one manifest line per shard in STAGE/manifest-gepnic.jsonl.
Re-runnable: a portal-year with a done marker is skipped.
  nice -n 15 .venv/bin/python export/gepnic_export.py STAGE [portal ...]
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SQL = HERE / "gepnic_export.sql"
SHARD_ROWS = 120_000          # bid rows in tenders with 2..60 sellers (the pair screens' load); larger tenders count 1/10
QUIET = (21, 9)               # start a new portal-year only between 21:00 and 09:00 IST
# validation portals first (preregistration-v2 section 6), then the rest by size
FIRST = [("gepnic-coalindiatenders", range(2013, 2019)), ("gepnic-govtprocurement-delhi", range(2011, 2016)),
         ("gepnic-defproc", range(2018, 2022)), ("gepnic-etenders-hry", range(2011, 2015))]

T_TYPES = {"tender_id": pa.string(), "yr": pa.int16(), "published": pa.string(), "closing": pa.string(),
           "ref_type": pa.string(), "status": pa.string(), "item_id": pa.int64(), "item_key": pa.string(), "buyer_id": pa.int64(),
           "ministry": pa.string(), "organisation": pa.string(), "state": pa.string(), "n_received": pa.int32(),
           "n_tech": pa.int32(), "n_priced": pa.int32(), "award_value": pa.float64(), "title": pa.string(), "org_id": pa.string(),
           "category": pa.string()}
B_TYPES = {"tender_id": pa.string(), "seller_id": pa.int64(), "seller_key": pa.string(), "tech": pa.int8(),
           "submitted_at": pa.string(), "price": pa.float64(), "rank_n": pa.int16(), "mse": pa.bool_(),
           "status": pa.string(), "stage": pa.string()}


def quiet_now():
    h = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).hour
    return h >= QUIET[0] or h < QUIET[1]


def wait_quiet():
    while not quiet_now():
        print(time.strftime("%H:%M"), "outside quiet hours: waiting", flush=True)
        time.sleep(600)


def psql(q, project, lo, hi, out):
    cmd = ["nice", "-n", "15", "sudo", "-n", "-u", "postgres", "psql", "-q", "-X", "-d", "scrape_hub", "-v", f"q={q}",
           "-v", f"project={project}", "-v", f"lo={lo}", "-v", f"hi={hi}"]
    with open(out, "wb") as f, open(SQL, "rb") as script:        # the script on stdin: postgres cannot read this home
        subprocess.run(cmd, stdin=script, stdout=f, check=True)


def read_csv(path, types):
    opts = pc.ConvertOptions(column_types=types, timestamp_parsers=["%Y-%m-%d %H:%M:%S"], strings_can_be_null=True)
    return pc.read_csv(path, convert_options=opts, read_options=pc.ReadOptions(block_size=1 << 26))


def parse_times(tab, cols):
    """GePNIC text times -> timestamp; an impossible date (31-Jun) or another format -> null."""
    import pandas as pd
    for c in cols:
        v = pd.to_datetime(tab[c].to_pandas(), format="%d-%b-%Y %I:%M %p", errors="coerce")
        tab = tab.set_column(tab.schema.get_field_index(c), c, pa.array(v, pa.timestamp("s")))
    return tab


def shard_plan(t, b):
    """org_id -> shard number: whole org_ids packed greedily up to SHARD_ROWS weighted bid rows."""
    import pandas as pd
    td = t.select(["tender_id", "org_id"]).to_pandas()
    n = b.select(["tender_id"]).to_pandas().value_counts("tender_id").rename("n").reset_index()
    td = td.merge(n, on="tender_id", how="left").fillna({"n": 0})
    td["w"] = td["n"].where(td["n"] <= 60, td["n"] / 10)
    w = td.groupby(td["org_id"].fillna(""))["w"].sum().sort_values(ascending=False)
    plan, loads = {}, []
    for org, x in w.items():
        k = next((i for i, l in enumerate(loads) if l + x <= SHARD_ROWS), None)
        if k is None:
            loads.append(0.0); k = len(loads) - 1
        loads[k] += x; plan[org] = k
    return plan


def export_year(stage, project, yr):
    portal = project.removeprefix("gepnic-")
    base = stage / "gepnic" / portal
    done = base / f".{yr}.done"
    if done.exists():
        return []
    base.mkdir(parents=True, exist_ok=True)
    tcsv, bcsv = base / f".{yr}.tenders.csv", base / f".{yr}.bids.csv"
    t0 = time.time()
    psql("tenders", project, str(yr), str(yr + 1), tcsv)
    psql("bids", project, str(yr), str(yr + 1), bcsv)
    secs = round(time.time() - t0, 1)
    t, b = read_csv(tcsv, T_TYPES), read_csv(bcsv, B_TYPES)
    t, b = parse_times(t, ["published", "closing"]), parse_times(b, ["submitted_at"])
    lines = []
    if t.num_rows:
        import pyarrow.compute as pcm
        plan = shard_plan(t, b)
        org = pcm.fill_null(t["org_id"], "")
        shard_of = pa.array([plan.get(o, 0) for o in org.to_pylist()], pa.int32())
        t = t.append_column("_shard", shard_of)
        tid_shard = dict(zip(t["tender_id"].to_pylist(), shard_of.to_pylist()))
        bshard = pa.array([tid_shard.get(x, -1) for x in b["tender_id"].to_pylist()], pa.int32())
        for k in sorted(set(plan.values())):
            sid = f"gepnic/{portal}/{yr}-{k}"
            d = stage / sid
            d.mkdir(parents=True, exist_ok=True)
            tk = t.filter(pcm.equal(t["_shard"], k)).drop_columns(["_shard"])
            bk = b.filter(pcm.equal(bshard, k))
            pq.write_table(tk, d / "tenders.parquet", compression="zstd")
            pq.write_table(bk, d / "bids.parquet", compression="zstd")
            files = {f: (d / f).stat().st_size for f in ("tenders.parquet", "bids.parquet")}
            lines.append({"shard": sid, "source": "gepnic", "portal": project, "year": yr, "n_tenders": tk.num_rows,
                          "n_bids": bk.num_rows, "files": files, "export_s": secs})
    tcsv.unlink(); bcsv.unlink()
    with open(stage / "manifest-gepnic.jsonl", "a") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    done.write_text(json.dumps({"tenders": t.num_rows, "bids": b.num_rows, "shards": len(lines), "export_s": secs}))
    print(time.strftime("%H:%M:%S"), project, yr, f"tenders {t.num_rows} bids {b.num_rows} shards {len(lines)} in {secs}s", flush=True)
    return lines


PORTALS = """etender-up eproc-rajasthan jktenders mahatenders wbtenders tntenders govtprocurement-delhi tendersodisha etenders-hry
hptenders etenders-kerala jharkhandtenders defproc mptenders uktenders etenders etenders-chd eproc-punjab assamtenders tripuratenders
iocletenders eprocure-andamannicobar coalindiatenders eprocurentpc tenders-ladakh eprocure-goa pudutenders arunachaltenders eprocurebhel
ddtenders tendersutl eprocuremdl nagalandtenders cpcletenders sikkimtender dnhtenders eprocurehsl eprocuregsl eprocurebel
meghalayatenders eprocuregrse mizoramtenders eprocuremidhani""".split()   # 43 GePNIC portals with bid lists (data-inventory.md), most OCR-priced tenders first


def portals():
    """Every portal, every tender-ID year 2007-2026 (an empty year is one short index range scan)."""
    return [("gepnic-" + p, range(2007, 2027)) for p in PORTALS]


def main():
    stage = Path(sys.argv[1]); stage.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[2:])
    todo = list(FIRST) + portals()
    seen = set()
    for project, years in todo:
        if only and project not in only:
            continue
        for yr in years:
            if (project, yr) in seen:
                continue
            seen.add((project, yr))
            wait_quiet()
            export_year(stage, project, yr)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
