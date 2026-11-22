#!/usr/bin/env python3
"""Pool task `screens`: compute the v1 and v2 bid screens of one shard at a time while >= 30 min of budget is left.

A queue item: {"shard": "gem/c0001", "source": "gem"|"gepnic", "files": [bucket keys], "directors": key or null, "code": sha}.
The task asks the hub for short-lived read links (no storage keys here), reads the Parquet inputs, runs
  v1 (screens/bidscreens_v1.py, frozen 2026-11-22: tender screens, pair screens + 200 placebo rounds, market rollup) and
  v2 (screens/v2.py, preregistration-v2: B-to-RA join, sync2, DQ, pairs with C3 + 50 placebo rounds, closed set, rollups),
uploads out/<shard>/*.parquet and records one summary per shard (counts and rates only, no firm names).
It refuses to run when the item's code hash differs from this checkout (the pre-registration pins the code).
"""
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screens import bidscreens_v1 as v1  # noqa: E402
from screens import fastpairs, labels, v2  # noqa: E402

fastpairs.install(v1)                      # speed only: the same pair rows as v1._pairs_of (tests/test_fastpairs.py)

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", "")
BUDGET_MIN = float(os.environ.get("SCRAPE_MAX_HOURS", "1")) * 60
START = time.time()
MIN_BUDGET_MIN = float(os.environ.get("BID_SCREENS_MIN_BUDGET_MIN", "30"))   # no new shard with less time left (a shard takes <= ~20 min)
V1_ROUNDS = 200
# preregistration-v2 A1: the hash before the tie12 bug fix; its definitions are the same, so its items run here too
A1_SAME = ("1ca7d9c802f6130c8143e65a4b66807280fad97935bfdc84b75213bb75aba90b",    # first freeze
           "9ec4735a53e9879006e0ca9fe41353b31cd05db7c605f1f3d3af4bfb64c89980")    # after A1


def code_sha():
    h = hashlib.sha256()
    for f in ("screens/bidscreens_v1.py", "screens/v2.py", "screens/labels.py", "screens/fastpairs.py", "run_shard.py"):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()


def read_inputs(paths):
    t = pd.read_parquet(paths["tenders.parquet"])
    b = pd.concat([pd.read_parquet(p) for n, p in sorted(paths.items()) if n.startswith("bids")], ignore_index=True)
    d = pd.read_parquet(paths["directors.parquet"]) if "directors.parquet" in paths else None
    return t, b, d


def run_v1(t, b, dirs, rounds=V1_ROUNDS):
    """The frozen v1 pipeline (bidscreens_v1.main tender / pair / market), on DataFrames instead of CSV files."""
    t = t.copy(); b = b.copy()
    for c in ("price", "rank_n", "tech", "seller_id"):
        b[c] = v1._num(b[c])
    b["submitted_at"] = v1._naive(b["submitted_at"])
    t["closing"] = v1._naive(t["closing"]); t["published"] = v1._naive(t["published"])
    keep_b = ["tender_id", "seller_id", "seller_key", "tech", "submitted_at", "price", "rank_n", "mse"]
    t, b, log = v1.clean(t, b[keep_b])
    if b.empty:                     # A2: no bid left after cleaning (only RA bids): v1 has nothing to screen here
        e = pd.DataFrame()
        return pd.DataFrame(columns=["tender_id"]), e, pd.DataFrame(columns=["tender_id", "a", "b"]), e, e, {**log, "empty": True}
    ts = v1.within_class_percentiles(v1.tender_screens(t, b))
    bp = b[b["tech"].isin([1, 2, 3]) | b["tech"].isna()]
    g = v1.pair_core(bp, t)
    links = v1.pair_core.last_links if len(g) else pd.DataFrame(columns=["tender_id", "a", "b"])
    pl = pd.DataFrame()
    if len(g):
        f = v1.firm_rates(bp)
        g = v1.shared_directors(g, dirs, f)
        g["a_key"] = g["a"].map(f["seller_key"]); g["b_key"] = g["b"].map(f["seller_key"])
        g["a_win_rate_all"] = g["a"].map(f["win_rate"]); g["b_win_rate_all"] = g["b"].map(f["win_rate"])
        g = v1.pair_flags(g)
        pl = v1.placebo_pairs(bp, t, rounds)
    m = v1.market_rollup(ts, g if len(g) else None, links if len(g) else None)
    return ts, g, links, pl, m, log


def run_v2(t, b, dirs, source):
    t, b = v2.prepare(t, b, source)
    link = v2.ra_join(t, b) if source == "gem" else pd.DataFrame(columns=["ra_id", "b_id", "n_ra"])
    te, be = v2.effective(t, b, link)
    n_ra = int((t["ref_type"].astype(str) != "B").sum())
    if be.empty:                    # A2: no bid on a B tender: nothing to screen
        e = pd.DataFrame()
        return (pd.DataFrame(columns=["tender_id", "yr", "source", "regulated"]), e, pd.DataFrame(columns=["tender_id", "a", "b"]), e,
                pd.DataFrame(columns=["n_tenders", "closed_flag"]), link, {"ra_tenders": n_ra, "ra_linked": int(len(link)), "b_tenders": int(len(te)), "empty": True})
    ts = v2.tender_v2(te, be)
    g, links, pl = v2.pairs_v2(te, be, dirs)
    mk, cst = v2.closed_set(te, be)
    ts = v2.finish(ts, g, links, cst)
    m = v2.market_v2(ts, mk)
    n_ra = int((t["ref_type"].astype(str) != "B").sum())
    return ts, g, links, pl, m, link, {"ra_tenders": n_ra, "ra_linked": int(len(link)), "b_tenders": int(len(te))}


def rate(s):
    s = pd.to_numeric(s, errors="coerce")
    n = int(s.notna().sum())
    return {"n": n, "flag": int((s == 1).sum()), "rate": round(float((s == 1).sum()) / n, 6) if n else None}


YEAR_COLS = ("single_bid", "cv_flag", "rd_flag", "pd_flag", "tie12_flag", "sync2_flag", "dq_flag", "dq2", "closed_t", "pair_v2_t")


def by_year(ts):
    """Per tender-year: tenders, and for each screen the number eligible and flagged (national base rates by year)."""
    out = {}
    for yr, g in ts.groupby("yr"):
        out[str(int(yr))] = {"tenders": int(len(g)), **{c: [int(pd.to_numeric(g[c], errors="coerce").notna().sum()),
                                                              int((pd.to_numeric(g[c], errors="coerce") == 1).sum())]
                                                          for c in YEAR_COLS if c in g}}
    return out


def summarize(shard, source, ts1, g1, pl1, ts2, g2, pl2, m2, extra, secs):
    s = {"shard": shard, "source": source, "code": code_sha(), "seconds": secs, **extra,
         "v1": {"tenders": int(len(ts1)), "pairs": int(len(g1)),
                "flags": {c: rate(ts1[c]) for c in ("single_bid", "cv_flag", "rd_flag", "pd_flag", "kurt_flag", "tie12_flag",
                                                    "sync_pairs_flag", "imhof_s1", "imhof_s3") if c in ts1},
                "pair_flags": {c: int(g1[c].sum()) for c in ("flag_rotation", "flag_sync", "flag_director", "flag_cover", "flag_any")
                               if len(g1) and c in g1},
                "placebo": pl1.drop(columns="round").agg(["mean", lambda x: x.quantile(0.95)]).set_axis(["mean", "p95"]).to_dict()
                if len(pl1) else {}},
         "v2": {"tenders": int(len(ts2)), "pairs": int(len(g2)),
                "flags": {c: rate(ts2[c]) for c in ("single_bid", "cv_flag", "rd_flag", "pd_flag", "kurt_flag", "tie12_flag",
                                                    "sync2_flag", "dq_flag", "dq2", "closed_t", "pair_v2_t") if c in ts2},
                "regulated_tenders": int(ts2["regulated"].astype(bool).sum()),
                "pair_flags": {c: int(g2[c].sum()) for c in ("flag_rotation", "flag_sync", "flag_director", "flag_cover",
                                                             "flag_cover_c3", "flag_any_v2") if len(g2) and c in g2},
                "placebo": pl2.drop(columns="round").agg(["mean", lambda x: x.quantile(0.95)]).set_axis(["mean", "p95"]).to_dict()
                if len(pl2) else {},
                "by_year": by_year(ts2),
                "markets": int((m2["n_tenders"] >= v2.MARKET_MIN_TENDERS).sum()),
                "closed_flag_markets": int((m2["closed_flag"] == 1).sum())}}
    return json.loads(json.dumps(s, default=lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)))


def to_parquet(df):
    buf = io.BytesIO()
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


V1_TENDER_COLS = ["tender_id", "yr", "item_id", "buyer_id", "n_submitted", "n_priced_q", "cv", "rd", "pd", "kurt", "tie12",
                  "sync_pairs", "single_bid", "imhof_s1", "imhof_s3", "norm_group", "cv_pct", "rd_pct", "pd_pct", "kurt_pct",
                  "cv_flag", "rd_flag", "pd_flag", "kurt_flag", "tie12_flag", "sync_pairs_flag", "price_screen_score"]


def process(p, paths):
    if paths is None:                                   # scrape-check / check workflow: a made-up shard
        from screens import synthetic
        t, b, dirs = synthetic.shard(int(p["synthetic"]))
    else:
        t, b, dirs = read_inputs(paths)
    t0 = time.time()
    ts1, g1, links1, pl1, m1, log1 = run_v1(t, b, dirs)
    t1 = time.time()
    ts2, g2, links2, pl2, m2, link, extra = run_v2(t, b, dirs, p["source"])
    t2 = time.time()
    lab = labels.label(ts2)
    if len(lab):
        v1c = ts1[[c for c in ("tender_id", "price_screen_score", "sync_pairs", "cv_flag", "rd_flag", "pd_flag", "tie12_flag",
                               "sync_pairs_flag") if c in ts1]].add_prefix("v1_").rename(columns={"v1_tender_id": "tender_id"})
        lab = lab.merge(ts2, on="tender_id", how="left").merge(v1c, on="tender_id", how="left")
    secs = {"v1": round(t1 - t0, 1), "v2": round(t2 - t1, 1)}
    summ = summarize(p["shard"], p["source"], ts1, g1, pl1, ts2, g2, pl2, m2, {**extra, "v1_clean": log1}, secs)
    files = {"v1_tender.parquet": ts1[[c for c in V1_TENDER_COLS if c in ts1]], "v1_market.parquet": m1, "v2_tender.parquet": ts2,
             "v2_market.parquet": m2, "labeled.parquet": lab}
    if len(g1):
        files["v1_pairs.parquet"] = g1[g1["flag_any"]]
        files["v1_placebo.parquet"] = pl1
    if len(g2):
        files["v2_pairs.parquet"] = g2[g2["flag_any_v2"] | g2["flag_cover"]]
        files["v2_pair_links.parquet"] = links2.merge(g2.loc[g2["flag_any_v2"] | g2["flag_cover"], ["a", "b"]], on=["a", "b"])
        files["v2_placebo.parquet"] = pl2
        files["v2_pairs_all_counts.parquet"] = g2[["n_joint", "rotation", "order_share", "sync_share", "n_shared_din"]].describe().reset_index()
    if len(link):
        files["v2_ra_link.parquet"] = link
    up = [(f"out/{p['shard']}/{n}", to_parquet(df), "application/octet-stream") for n, df in files.items()]
    up.append((f"out/{p['shard']}/summary.json", json.dumps(summ).encode(), "application/json"))
    return summ, up


def fetch(hub, session, keys, tmp):
    urls = hub.read_urls(PROJECT, keys)
    out = {}
    for k in keys:
        t = time.time()
        r = session.get(urls[k], timeout=(30, 600))
        hub.metric("n_http"); hub.metric("t_http_s", time.time() - t)
        if r.status_code != 200:
            hub.metric(f"http_{r.status_code}")
            r.raise_for_status()
        f = Path(tmp) / k.replace("/", "__")
        f.write_bytes(r.content)
        out[k.rsplit("/", 1)[1]] = f
    return out


def main():
    from scrape_hub_client import HubClient
    hub = HubClient(PROJECT, job_index=int(os.environ.get("SCRAPE_JOB_INDEX", 0)), total_jobs=int(os.environ.get("SCRAPE_TOTAL_JOBS", 1)))
    s = requests.Session()
    sha = code_sha()
    try:
        while BUDGET_MIN - (time.time() - START) / 60 >= MIN_BUDGET_MIN:
            if hub.settings.get("paused"):
                time.sleep(30); continue
            items = hub.claim(1, steal=True, retry_failed=True)
            if not items:
                if hub.settings.get("paused"):
                    time.sleep(5); continue
                break
            it = items[0]; p = it["payload"]
            if p.get("code") not in (sha, *A1_SAME) and not p.get("synthetic"):
                hub.complete([it["key"]], "failed", error=f"code {sha[:12]} differs from the pinned {str(p.get('code'))[:12]}")
                hub.metric("code_mismatch"); break
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    keys = list(p.get("files") or []) + ([p["directors"]] if p.get("directors") else [])
                    paths = fetch(hub, s, keys, tmp) if keys else None
                    if paths is None:
                        hub.metric("n_http"); hub.metric("t_http_s", 0.001)   # nothing fetched (made-up shard)
                    summ, up = process(p, paths)
                hub.upload_many(up)
                hub.record(it["key"], summ)
                hub.metric("shards_done")
                hub.complete([it["key"]], "done", fetched=1)
            except Exception as e:
                print("failed", it["key"], repr(e)[:300], file=sys.stderr)
                hub.complete([it["key"]], "failed", error=repr(e)[:200])
        hub.close()
    except BaseException as e:
        hub.close("failed", f"{type(e).__name__}: {e}"[:200])
        raise


if __name__ == "__main__":
    main()
