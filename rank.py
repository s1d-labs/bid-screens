#!/usr/bin/env python3
"""Pool task `rank`: the national step after every shard is done (one queue item per run).

Item payload: {"run": "...", "shards": [{"project": "bid-screens-gem", "shard": "gem/c0001"}, ...], "code": sha}.
Reads each shard's summary.json, v1/v2 market files, labeled tenders and flagged pairs (read links from the hub), and writes
national/<run>/: base_rates.json, markets_v2.parquet (composite_v2), markets_v1.parquet (v1 composite re-ranked nationally),
labeled.parquet, validation.json (preregistration-v2 section 6), pairs_v2.parquet, leads.parquet (section 7).
Every output is an indicator; none of it is published.
"""
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from screens import v2  # noqa: E402
from screens.labels import CASES  # noqa: E402

PROJECT = os.environ.get("SCRAPE_POOL_PROJECT", "")
SEED, BOOT = 20261122, 2000
V1_COMPS = ["price_score", "tie12_rate", "sync_rate", "flagged_pair_rate"]


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan, 0.0
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), float(len(pos) * len(neg))


def pooled_auc(groups):
    """groups: list of (pos, neg) per case -> within-case pooled AUC (weights = pairs compared)."""
    num = den = 0.0
    for p, n in groups:
        a, w = auc(p, n)
        if np.isfinite(a):
            num += a * w; den += w
    return num / den if den else np.nan


def boot_ci(groups, rng):
    vals = []
    for _ in range(BOOT):
        g2 = [(rng.choice(p, len(p)) if len(p) else p, rng.choice(n, len(n)) if len(n) else n) for p, n in groups]
        vals.append(pooled_auc(g2))
    vals = np.array(vals, float)
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))) if np.isfinite(vals).any() else (None, None)


def validation(lab, markets):
    rng = np.random.default_rng(SEED)
    out = {"tender_level": {}, "market_level": {}}
    scores = ["tender_v2", "price_rank", "sync2_pct", "dq_pct", "closed_t", "pair_v2_t", "v1_price_screen_score", "v1_sync_pairs"]
    for sc in scores:
        if sc not in lab:
            continue
        res = {"cases": {}}
        groups = []
        for case, c in CASES.items():
            x = lab[lab["case"] == case]
            p = pd.to_numeric(x.loc[x["label"] == 1, sc], errors="coerce").dropna().values
            n = pd.to_numeric(x.loc[x["label"] == 0, sc], errors="coerce").dropna().values
            a, _ = auc(p, n)
            lo, hi = boot_ci([(p, n)], rng) if len(p) and len(n) else (None, None)
            res["cases"][case] = {"n_pos": int(len(p)), "n_ctrl": int(len(n)), "auc": None if not np.isfinite(a) else float(a),
                                  "ci": [lo, hi], "too_few": bool(len(p) < 5), "primary": c["primary"]}
            if c["primary"]:
                groups.append((p, n))
        pa = pooled_auc(groups)
        lo, hi = boot_ci(groups, rng)
        res["pooled_primary"] = {"auc": None if not np.isfinite(pa) else float(pa), "ci": [lo, hi],
                                 "n_pos": int(sum(len(p) for p, _ in groups)), "n_ctrl": int(sum(len(n) for _, n in groups)),
                                 "success": bool(lo is not None and lo > 0.5)}
        out["tender_level"][sc] = res
    # market level: market-years holding >= 1 positive vs >= 1 control (>= 10 tenders)
    lab = lab.assign(m=lab["buyer_id"].astype(str) + "|" + lab["item_id"].astype(str) + "|" + lab["yr"].astype(str))
    mk = markets.set_index("m")
    for sc in ("composite_v2", "closed_excess", "closed_share", "sync2_rate", "dq_rate", "price_rank", "pair_v2_rate"):
        groups, per = [], {}
        for case, c in CASES.items():
            x = lab[lab["case"] == case]
            pm = set(x.loc[x["label"] == 1, "m"]) & set(mk.index)
            nm = (set(x.loc[x["label"] == 0, "m"]) - pm) & set(mk.index)
            p = mk.loc[list(pm), sc].dropna().values if pm else np.array([])
            n = mk.loc[list(nm), sc].dropna().values if nm else np.array([])
            a, _ = auc(p, n)
            per[case] = {"n_pos_markets": int(len(p)), "n_ctrl_markets": int(len(n)), "auc": None if not np.isfinite(a) else float(a),
                         "pos_percentile_among_ctrl": [float((n < v).mean() + 0.5 * (n == v).mean()) for v in p] if len(n) else []}
            if c["primary"]:
                groups.append((p, n))
        npos = sum(len(p) for p, _ in groups)
        pa = pooled_auc(groups) if npos >= 5 else np.nan
        ci = boot_ci(groups, rng) if npos >= 5 else (None, None)
        out["market_level"][sc] = {"cases": per, "pooled_primary": {"auc": None if not np.isfinite(pa) else float(pa), "ci": list(ci),
                                                                   "n_pos": int(npos), "claimed": bool(npos >= 5)}}
    return out


def base_rates(summaries):
    out = {}
    for s in summaries:
        src = s["source"]
        o = out.setdefault(src, {"shards": 0, "v1": {}, "v2": {}, "v1_pairs": {}, "v2_pairs": {}, "by_year": {}, "extra": {}})
        o["shards"] += 1
        for ver in ("v1", "v2"):
            for k, r in s[ver]["flags"].items():
                a = o[ver].setdefault(k, [0, 0]); a[0] += r["n"]; a[1] += r["flag"]
            o[ver].setdefault("_tenders", [0, 0])[0] += s[ver]["tenders"]
            o[ver + "_pairs"]["pairs"] = o[ver + "_pairs"].get("pairs", 0) + s[ver]["pairs"]
            for k, n in s[ver]["pair_flags"].items():
                o[ver + "_pairs"][k] = o[ver + "_pairs"].get(k, 0) + n
            for k, d in (s[ver].get("placebo") or {}).items():
                for stat in ("mean", "p95"):
                    o[ver + "_pairs"].setdefault(f"placebo_{k}_{stat}_sum", 0.0)
                    o[ver + "_pairs"][f"placebo_{k}_{stat}_sum"] += float(d.get(stat) or 0)
        for yr, d in s["v2"].get("by_year", {}).items():
            y = o["by_year"].setdefault(yr, {})
            for k, v in d.items():
                if isinstance(v, list):
                    a = y.setdefault(k, [0, 0]); a[0] += v[0]; a[1] += v[1]
                else:
                    y[k] = y.get(k, 0) + v
        for k in ("ra_tenders", "ra_linked", "b_tenders"):
            o["extra"][k] = o["extra"].get(k, 0) + int(s.get(k) or 0)
        o["extra"]["regulated_tenders"] = o["extra"].get("regulated_tenders", 0) + s["v2"]["regulated_tenders"]
        o["extra"]["closed_flag_markets"] = o["extra"].get("closed_flag_markets", 0) + s["v2"]["closed_flag_markets"]
        o["extra"]["markets_ge10"] = o["extra"].get("markets_ge10", 0) + s["v2"]["markets"]
        o["extra"]["cpu_s"] = o["extra"].get("cpu_s", 0) + sum(s["seconds"].values())
    return out


def leads(m2, pairs):
    m = m2[(~m2["regulated"].astype(bool)) & (m2["n_tenders"] >= v2.MARKET_MIN_TENDERS)].copy()
    m["top1"] = m.groupby("source")["composite_v2"].rank(pct=True) >= 0.99
    top = m[m["top1"]].sort_values("composite_v2", ascending=False).drop_duplicates(["source", "buyer_id"])
    p = pairs.copy()
    if len(p):
        p["n_flags"] = p[["flag_rotation", "flag_sync", "flag_director", "flag_cover_c3"]].sum(axis=1)
        p = p[(p["n_flags"] >= 2) | p["flag_cover_c3"] | p["flag_rotation"]]
    return top, p


def get(hub, session, keys):
    out = {}
    for i in range(0, len(keys), 100):
        urls = hub.read_urls(PROJECT, keys[i:i + 100])
        for k, u in urls.items():
            t = time.time()
            r = session.get(u, timeout=(30, 300))
            hub.metric("n_http"); hub.metric("t_http_s", time.time() - t)
            out[k] = r.content if r.status_code == 200 else None
            if r.status_code != 200:
                hub.metric(f"http_{r.status_code}")
    return out


def to_pq(df):
    buf = io.BytesIO()
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string")
    df.to_parquet(buf, index=False, compression="zstd")
    return buf.getvalue()


def national(blobs):
    """blobs: {shard: {name: bytes}} -> (files to upload, small json results)."""
    rd = lambda x: pd.read_parquet(io.BytesIO(x)) if x else pd.DataFrame()
    summ = [json.loads(b["summary.json"]) for b in blobs.values() if b.get("summary.json")]
    m2 = pd.concat([rd(b.get("v2_market.parquet")) for b in blobs.values()], ignore_index=True)
    m1 = pd.concat([rd(b.get("v1_market.parquet")).assign(source=s.split("/")[0]) for s, b in blobs.items()], ignore_index=True)
    lab = pd.concat([rd(b.get("labeled.parquet")) for b in blobs.values()], ignore_index=True)
    pairs = pd.concat([rd(b.get("v2_pairs.parquet")).assign(shard=s) for s, b in blobs.items()], ignore_index=True)
    m2 = m2.groupby("m", as_index=False).first() if len(m2) else m2      # a split class's RA can repeat a market: keep one
    m2 = v2.national_composite(m2)
    if len(m1):
        for c in V1_COMPS:
            m1[c + "_rn"] = m1.groupby("source")[c].rank(pct=True)
        m1["composite_nat"] = m1[[c + "_rn" for c in V1_COMPS]].mean(axis=1, skipna=True)
    lab = lab.drop_duplicates(["case", "tender_id"])
    val = validation(lab, m2) if len(lab) else {}
    top, lp = leads(m2, pairs)
    files = {"markets_v2.parquet": m2, "markets_v1.parquet": m1, "labeled.parquet": lab, "pairs_v2.parquet": pairs,
             "leads_markets.parquet": top, "leads_pairs.parquet": lp}
    js = {"base_rates.json": base_rates(summ), "validation.json": val,
          "counts.json": {"shards": len(blobs), "summaries": len(summ), "markets_v2": int(len(m2)), "labeled": int(len(lab)),
                          "pairs_v2": int(len(pairs)), "lead_markets": int(len(top)), "lead_pairs": int(len(lp))}}
    return files, js


NAMES = ("summary.json", "v1_market.parquet", "v2_market.parquet", "labeled.parquet", "v2_pairs.parquet")


def main():
    from scrape_hub_client import HubClient
    hub = HubClient(PROJECT, job_index=int(os.environ.get("SCRAPE_JOB_INDEX", 0)), total_jobs=int(os.environ.get("SCRAPE_TOTAL_JOBS", 1)))
    s = requests.Session()
    try:
        items = hub.claim(1, steal=True, retry_failed=True)
        for it in items:
            p = it["payload"]
            try:
                keys = [f"{x['project']}/out/{x['shard']}/{n}" for x in p["shards"] for n in NAMES]
                # read links are per project: ask in groups of one project
                got = {}
                for proj in sorted({x["project"] for x in p["shards"]}):
                    ks = [k for k in keys if k.startswith(proj + "/")]
                    for i in range(0, len(ks), 100):              # sign and fetch one batch at a time (links expire)
                        for k, u in hub.read_urls(proj, ks[i:i + 100]).items():
                            for attempt in range(3):
                                t = time.time(); r = s.get(u, timeout=(30, 300))
                                hub.metric("n_http"); hub.metric("t_http_s", time.time() - t)
                                if r.status_code in (200, 404):
                                    break
                                hub.metric(f"http_{r.status_code}")
                                u = hub.read_urls(proj, [k])[k]
                            got[k] = r.content if r.status_code == 200 else None
                    missing = [k for k in ks if k.endswith("summary.json") and got.get(k) is None]
                    if missing:
                        raise RuntimeError(f"{len(missing)} shard summaries of {proj} could not be read, e.g. {missing[0]}")
                blobs = {}
                for x in p["shards"]:
                    blobs[x["shard"]] = {n: got.get(f"{x['project']}/out/{x['shard']}/{n}") for n in NAMES}
                files, js = national(blobs)
                up = [(f"national/{p['run']}/{n}", to_pq(df), "application/octet-stream") for n, df in files.items()]
                up += [(f"national/{p['run']}/{n}", json.dumps(v, default=float).encode(), "application/json") for n, v in js.items()]
                hub.upload_many(up)
                hub.record(it["key"], {"run": p["run"], **js["counts.json"]})
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
