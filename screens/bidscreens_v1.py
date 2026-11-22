#!/usr/bin/env python3
"""Bet 3: bid-rigging screens with controls and placebos. Read-only; numpy + pandas only.

Inputs (CSV, from gem_export.sql or the national Parquet export with the same columns):
  tenders.csv  tender_id, yr, published, closing, ref_type, status, item_id, item_key, buyer_id, ministry, organisation,
               state, n_received, n_tech, n_priced, award_value
  bids.csv     tender_id, seller_id, seller_key, tech, submitted_at, price, rank_n, mse
  directors.csv (optional) seller_key, din    -- one row per seller and director (current directors, no dates)

Commands:
  bidscreens.py tender  TENDERS BIDS OUT            per-tender screens + within-class percentiles  -> OUT/tender_screens.csv
  bidscreens.py pair    TENDERS BIDS OUT [--dir D]  firm-pair screens + permutation placebo         -> OUT/pair_screens.csv
  bidscreens.py market  OUT                         buyer-market rollup of tender + pair flags      -> OUT/market_screens.csv
  bidscreens.py validate OUT LABELS.csv             AUC (bootstrap CI) of market/pair scores; LABELS: unit_type,unit_id,label
Every definition and threshold here is fixed in ../preregistration.md (frozen before the pilot ran). Do not tune them on
pilot outcomes; a change needs a dated amendment in preregistration.md.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---- pre-registered constants (preregistration.md section 3) ----
MIN_PRICED_DIST = 3        # CV, PD, RD, RDalt, spread need >= 3 priced bids (RD needs >= 2 losers)
MIN_PRICED_MOM = 4         # skewness, kurtosis need >= 4 priced bids
CLASS_MIN_TENDERS = 30     # a class-year with fewer tenders gets no percentile (pooled to class only)
FLAG_PCT = 0.05            # a tender screen flags when it sits in the most collusion-like 5 % of its class-year
SYNC_MIN = 10              # two submissions within 10 minutes = synchronous
SYNC_EARLY_MIN = 120       # ... only if both are >= 120 minutes before the closing time (deadline rush excluded)
PAIR_MIN_JOINT = 5         # firm-pair screens need >= 5 joint tenders (both submitted)
PERMUTATIONS = 200         # label-permutation placebo rounds for pair screens
ROT_MIN = 0.8              # pair takes all joint tenders and splits wins evenly (rotation index >= 0.8)
SYNC_SHARE_MIN = 0.5       # >= half of joint tenders with synchronous early submission
COVER_MAX_WINRATE = 0.05   # the always-higher firm wins <= 5 % of all its bids in scope
MARKET_MIN_TENDERS = 10    # a buyer-market needs >= 10 tenders in scope
SEED = 20261122
BOOT = 2000                # bootstrap rounds for AUC CI

# direction: +1 = high value is collusion-like, -1 = low value is collusion-like (literature.md, section 2)
DIRECTION = {"cv": -1, "pd": +1, "spread": -1, "rd": +1, "rdalt": +1, "kurt": +1, "skew": -1,
             "tie12": +1, "sync_pairs": +1}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _naive(col):
    """Parse times; tz-aware values (GePNIC export, +05:30) become naive Asia/Kolkata wall time."""
    x = pd.to_datetime(col, errors="coerce", utc=False, format="mixed")
    if getattr(x.dt, "tz", None) is not None:
        x = x.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return x


def load(tenders_csv, bids_csv):
    t = pd.read_csv(tenders_csv, low_memory=False)
    b = pd.read_csv(bids_csv, low_memory=False)
    for c in ("price", "rank_n", "tech", "seller_id"):
        b[c] = _num(b[c])
    b["submitted_at"] = _naive(b["submitted_at"])
    t["closing"] = _naive(t["closing"])
    t["published"] = _naive(t["published"])
    return t, b


def clean(t, b):
    """Pre-registered cleaning (preregistration.md 2.2). Returns (tenders, bids, log)."""
    log = {"tenders_in": len(t), "bids_in": len(b)}
    t = t[t["ref_type"].astype(str) != "R"]                       # reverse auctions apart
    log["after_drop_reverse_auction"] = len(t)
    t = t[~t["item_key"].astype(str).str.contains("automationtestcategory", case=False, na=False)]
    log["after_drop_test_items"] = len(t)
    b = b[b["tender_id"].isin(t["tender_id"])]
    b = b[~b["seller_key"].astype(str).str.fullmatch("TESTING", case=False, na=False)]
    # one row per seller per tender: keep the lowest positive price, the earliest submission
    b = (b.sort_values(["tender_id", "seller_id", "price"])
           .groupby(["tender_id", "seller_id"], as_index=False)
           .agg(seller_key=("seller_key", "first"), tech=("tech", "min"), submitted_at=("submitted_at", "min"),
                price=("price", "min"), rank_n=("rank_n", "min"), mse=("mse", "first")))
    log["bids_out"] = len(b)
    log["tenders_out"] = len(t)
    return t, b, log


def _moments(p):
    n = len(p)
    m = p.mean()
    s = p.std(ddof=1)
    if s == 0 or n < MIN_PRICED_MOM:
        return np.nan, np.nan
    z = (p - m) / s
    skew = n / ((n - 1) * (n - 2)) * np.sum(z ** 3)
    kurt = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(z ** 4)
            - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    return skew, kurt


def tender_stats(prices):
    """prices: 1-d array of priced, qualified bids of one tender (one per seller). Imhof-style screens."""
    p = np.sort(np.asarray([x for x in prices if x is not None and np.isfinite(x) and x > 0], dtype=float))
    out = {"n_priced_q": len(p)}
    if len(p) >= 2:
        out["tie12"] = float(p[0] == p[1])
        out["n_ident_max"] = int(pd.Series(p).value_counts().max())
    if len(p) >= MIN_PRICED_DIST:
        b1, b2 = p[0], p[1]
        losers = p[1:]
        out["cv"] = p.std(ddof=1) / p.mean()
        out["pd"] = (b2 - b1) / b1
        out["spread"] = (p[-1] - b1) / b1
        sl = losers.std(ddof=1)
        out["rd"] = (b2 - b1) / sl if sl > 0 else np.nan
        mean_gap_losers = (p[-1] - b2) / (len(p) - 2)        # mean consecutive gap among losing bids
        out["rdalt"] = (b2 - b1) / mean_gap_losers if mean_gap_losers > 0 else np.nan
    if len(p) >= MIN_PRICED_MOM:
        out["skew"], out["kurt"] = _moments(p)
    return out


def sync_pairs(sub, closing):
    """Count seller pairs that submitted within SYNC_MIN minutes, both >= SYNC_EARLY_MIN before closing."""
    s = sub.dropna().sort_values()
    if pd.notna(closing):
        s = s[s <= closing - pd.Timedelta(minutes=SYNC_EARLY_MIN)]
    v = s.values.astype("datetime64[s]").astype(np.int64) / 60.0
    n = 0
    for i in range(len(v)):
        j = i + 1
        while j < len(v) and v[j] - v[i] <= SYNC_MIN:
            n += 1
            j += 1
    return n


def tender_screens(t, b):
    q = b[(b["tech"] == 1) & (b["price"] > 0)]
    recs = [dict(tender_id=tid, **tender_stats(g.values)) for tid, g in q.groupby("tender_id")["price"]]
    stats = pd.DataFrame(recs).set_index("tender_id") if recs else pd.DataFrame(index=pd.Index([], name="tender_id"))
    nsub = b.groupby("tender_id").size().rename("n_submitted")
    close = t.set_index("tender_id")["closing"]
    sy = b.merge(close.rename("closing"), left_on="tender_id", right_index=True)
    sy = sy.groupby("tender_id").apply(lambda g: sync_pairs(g["submitted_at"], g["closing"].iloc[0]),
                                       include_groups=False).rename("sync_pairs")
    out = t.set_index("tender_id")[["yr", "item_id", "item_key", "buyer_id", "organisation", "ministry", "state",
                                    "n_received", "n_tech", "n_priced", "award_value", "status"]].copy()
    out = out.join(nsub).join(stats).join(sy)
    out["n_submitted"] = out["n_submitted"].fillna(0)
    if "cv" in out and "rd" in out:                     # secondary absolute rules (Imhof, Karagok & Rutz 2018)
        ok = out["cv"].notna() & out["rd"].notna()
        out["imhof_s1"] = ((out["cv"] <= 0.06) & (out["rd"] > 1.00)).astype(float).where(ok)
        out["imhof_s3"] = ((out["cv"] <= 0.03) & (out["rd"] > 1.30)).astype(float).where(ok)
    out["single_bid"] = (out["n_submitted"] == 1).astype(float)
    out.loc[out["n_submitted"] == 0, "single_bid"] = np.nan
    return out.reset_index()


def within_class_percentiles(ts):
    """Percentile of each screen within item class x year (pre-registered normalisation); collusion-like = high pct."""
    ts = ts.copy()
    size = ts.groupby(["item_id", "yr"])["tender_id"].transform("size")
    ts["norm_group"] = np.where(size >= CLASS_MIN_TENDERS, ts["item_id"].astype(str) + "|" + ts["yr"].astype(str),
                                "item|" + ts["item_id"].astype(str))
    gsize = ts.groupby("norm_group")["tender_id"].transform("size")
    ts.loc[gsize < CLASS_MIN_TENDERS, "norm_group"] = np.nan          # too small even pooled: no percentile
    for c, d in DIRECTION.items():
        if c not in ts:
            continue
        x = ts[c] * d
        ts[c + "_pct"] = x.groupby(ts["norm_group"]).rank(pct=True, method="average")
        ts[c + "_flag"] = (ts[c + "_pct"] >= 1 - FLAG_PCT).astype(float).where(ts[c + "_pct"].notna())
    core = [c + "_pct" for c in ("cv", "rd", "kurt", "pd")]
    ts["price_screen_score"] = ts[core].mean(axis=1, skipna=False) if all(c in ts for c in core) else np.nan
    return ts


# ---------------- pair screens ----------------
def _pairs_of(df):
    """All unordered seller pairs that co-submitted in a tender: tender_id, a, b (a < b)."""
    rows = []
    for tid, g in df.groupby("tender_id"):
        s = np.unique(g["seller_id"].values)
        if len(s) < 2 or len(s) > 60:                                 # very large bids: skip (preregistration 3.3)
            continue
        ii, jj = np.triu_indices(len(s), 1)
        rows.append(pd.DataFrame({"tender_id": tid, "a": s[ii], "b": s[jj]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["tender_id", "a", "b"])


def pair_core(b, t):
    """Joint counts, order persistence, win sharing, sync per firm pair."""
    winners = (b[(b["tech"] == 1) & (b["rank_n"] == 1)].groupby("tender_id")["seller_id"]
               .agg(lambda s: s.iloc[0] if len(s) == 1 else -1))          # -1 = tied L1 (no single winner)
    price = b.set_index(["tender_id", "seller_id"])["price"]
    sub = b.set_index(["tender_id", "seller_id"])["submitted_at"]
    close = t.set_index("tender_id")["closing"]
    pr = _pairs_of(b)
    if pr.empty:
        return pr
    ka = pd.MultiIndex.from_arrays([pr["tender_id"], pr["a"]]); kb = pd.MultiIndex.from_arrays([pr["tender_id"], pr["b"]])
    pa, pb = price.reindex(ka).values, price.reindex(kb).values
    sa, sb = sub.reindex(ka).values, sub.reindex(kb).values
    cl = close.reindex(pr["tender_id"]).values
    w = winners.reindex(pr["tender_id"]).values
    rank = b.set_index(["tender_id", "seller_id"])["rank_n"]
    ra, rb = rank.reindex(ka).values, rank.reindex(kb).values
    priced = np.isfinite(pa) & np.isfinite(pb) & (pa > 0) & (pb > 0)
    ranked = np.isfinite(ra) & np.isfinite(rb)
    pr["both_priced"] = priced | ranked                        # ordered by price, else by rank (GePNIC lists)
    pr["a_lower"] = np.where(priced, pa < pb, ranked & (ra < rb))
    pr["b_lower"] = np.where(priced, pb < pa, ranked & (rb < ra))
    pr["a_win"] = w == pr["a"].values
    pr["b_win"] = w == pr["b"].values
    dt = np.abs((pd.to_datetime(sa) - pd.to_datetime(sb)).total_seconds().values) / 60.0
    early = ((pd.to_datetime(cl) - np.maximum(pd.to_datetime(sa), pd.to_datetime(sb))).total_seconds().values / 60.0
             >= SYNC_EARLY_MIN)
    pr["sync"] = (dt <= SYNC_MIN) & early
    g = pr.groupby(["a", "b"]).agg(n_joint=("tender_id", "size"), n_both_priced=("both_priced", "sum"),
                                   a_lower=("a_lower", "sum"), b_lower=("b_lower", "sum"),
                                   a_wins=("a_win", "sum"), b_wins=("b_win", "sum"), n_sync=("sync", "sum"))
    g = g[g["n_joint"] >= PAIR_MIN_JOINT].reset_index()
    pair_core.last_links = pr.merge(g[["a", "b"]], on=["a", "b"])[["tender_id", "a", "b"]]
    nbp = g["n_both_priced"].where(g["n_both_priced"] > 0)
    g["order_share"] = np.maximum(g["a_lower"], g["b_lower"]) / nbp             # fixed price order
    g["pair_win_share"] = (g["a_wins"] + g["b_wins"]) / g["n_joint"]            # the two take the joint tenders
    g["balance"] = np.minimum(g["a_wins"], g["b_wins"]) / (g["a_wins"] + g["b_wins"]).where(lambda x: x > 0)
    g["rotation"] = g["pair_win_share"] * 2 * g["balance"]                      # 1 = they take all and split evenly
    g["sync_share"] = g["n_sync"] / g["n_joint"]
    return g


def pair_flags(g):
    """Pre-registered pair flags (preregistration.md 3.3)."""
    g["flag_rotation"] = (g["rotation"] >= ROT_MIN) & (g["n_joint"] >= PAIR_MIN_JOINT)
    g["flag_sync"] = (g["sync_share"] >= SYNC_SHARE_MIN) & (g["n_joint"] >= PAIR_MIN_JOINT)
    g["flag_director"] = g["n_shared_din"].fillna(0) >= 1
    lo_wr = np.where(g["a_lower"] >= g["b_lower"], g["b_win_rate_all"], g["a_win_rate_all"])
    g["flag_cover"] = ((g["order_share"] >= 1.0) & (g["n_both_priced"] >= PAIR_MIN_JOINT) & (lo_wr <= COVER_MAX_WINRATE))
    g["flag_any"] = g[["flag_rotation", "flag_sync", "flag_director", "flag_cover"]].any(axis=1)
    return g


def market_rollup(ts, g, links):
    """Buyer-market = buyer_id x item_id over the scope period. Components and composite (preregistration.md 3.4)."""
    ts = ts.copy()
    for c in ("cv", "cv_flag", "rd_flag", "tie12", "price_screen_score"):      # scopes without prices (GePNIC lists)
        if c not in ts:
            ts[c] = np.nan
    ts["sync_any"] = (ts["sync_pairs"] > 0).astype(float).where(ts["sync_pairs"].notna())
    if g is not None and len(g):
        fl = links.merge(g.loc[g["flag_any"], ["a", "b"]], on=["a", "b"])
        ts["flagged_pair"] = ts["tender_id"].isin(fl["tender_id"]).astype(float)
        for c in ("flag_rotation", "flag_sync", "flag_director", "flag_cover"):
            f2 = links.merge(g.loc[g[c], ["a", "b"]], on=["a", "b"])
            ts[c] = ts["tender_id"].isin(f2["tender_id"]).astype(float)
    m = ts.groupby(["buyer_id", "item_id"]).agg(
        organisation=("organisation", "first"), item_key=("item_key", "first"), n_tenders=("tender_id", "size"),
        n_dist=("cv", lambda x: int(x.notna().sum())), single_rate=("single_bid", "mean"),
        price_score=("price_screen_score", "mean"), cv_flag_rate=("cv_flag", "mean"), rd_flag_rate=("rd_flag", "mean"),
        tie12_rate=("tie12", "mean"), sync_rate=("sync_any", "mean"),
        flagged_pair_rate=("flagged_pair", "mean") if "flagged_pair" in ts else ("tie12", lambda x: np.nan),
        value=("award_value", "sum")).reset_index()
    m = m[m["n_tenders"] >= MARKET_MIN_TENDERS].copy()
    comps = ["price_score", "tie12_rate", "sync_rate", "flagged_pair_rate"]
    for c in comps:
        m[c + "_r"] = m[c].rank(pct=True)
    m["composite"] = m[[c + "_r" for c in comps]].mean(axis=1, skipna=True)
    m["market_id"] = m["buyer_id"].astype(str) + "|" + m["item_id"].astype(str)
    return m.sort_values("composite", ascending=False)


def firm_rates(b):
    f = b.groupby("seller_id").agg(seller_key=("seller_key", "first"), n_bids=("tender_id", "nunique"),
                                   n_wins=("rank_n", lambda r: int((r == 1).sum())))
    f["win_rate"] = f["n_wins"] / f["n_bids"]
    return f


def placebo_pairs(b, t, rounds=PERMUTATIONS):
    """Label-permutation placebo: shuffle seller labels within each item class x year, keep prices and times.
    Any pair score that is as high in the placebo is market structure, not a pair pattern."""
    rng = np.random.default_rng(SEED)
    key = t.set_index("tender_id")[["item_id", "yr"]]
    bb = b.join(key, on="tender_id")
    out = []
    for r in range(rounds):
        x = bb.copy()
        x["seller_id"] = x.groupby(["item_id", "yr"])["seller_id"].transform(lambda s: rng.permutation(s.values))
        x = x.drop_duplicates(["tender_id", "seller_id"])
        g = pair_core(x, t)
        ok = len(g) > 0
        out.append({"round": r, "pairs": len(g),
                    "rotation_ge_0_8": int((g["rotation"] >= 0.8).sum()) if ok else 0,
                    "order_share_1": int(((g["order_share"] == 1) & (g["n_both_priced"] >= PAIR_MIN_JOINT)).sum()) if ok else 0,
                    "sync_share_ge_0_5": int((g["sync_share"] >= 0.5).sum()) if ok else 0})
    return pd.DataFrame(out)


def shared_directors(pairs, dirs, f):
    """pairs: a, b seller_ids; dirs: seller_key, din. Adds n_shared_din (current directors; not dated)."""
    if dirs is None or dirs.empty:
        pairs["n_shared_din"] = np.nan
        return pairs
    k = f["seller_key"]
    d = dirs.groupby("seller_key")["din"].apply(set)
    sa = pairs["a"].map(k).map(d); sb = pairs["b"].map(k).map(d)
    pairs["a_has_cin"] = sa.notna(); pairs["b_has_cin"] = sb.notna()
    pairs["n_shared_din"] = [len(x & y) if isinstance(x, set) and isinstance(y, set) else np.nan for x, y in zip(sa, sb)]
    return pairs


# ---------------- validation ----------------
def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank(method="average").values
    rp = ranks[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(pos, neg, boot=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    a = auc(pos, neg)
    if not np.isfinite(a):
        return a, np.nan, np.nan
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(boot)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return a, lo, hi


def main():
    cmd = sys.argv[1]
    if cmd == "tender":
        t, b = load(sys.argv[2], sys.argv[3]); out = Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
        t, b, log = clean(t, b)
        ts = within_class_percentiles(tender_screens(t, b))
        ts.to_csv(out / "tender_screens.csv", index=False)
        json.dump(log, open(out / "clean_log.json", "w"), indent=1)
        print(json.dumps(log))
    elif cmd == "pair":
        t, b = load(sys.argv[2], sys.argv[3]); out = Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
        t, b, _ = clean(t, b)
        b = b[b["tech"].isin([1, 2, 3]) | b["tech"].isna()]
        g = pair_core(b, t)
        f = firm_rates(b)
        dirs = pd.read_csv(sys.argv[sys.argv.index("--dir") + 1]) if "--dir" in sys.argv else None
        g = shared_directors(g, dirs, f)
        g["a_key"] = g["a"].map(f["seller_key"]); g["b_key"] = g["b"].map(f["seller_key"])
        g["a_win_rate_all"] = g["a"].map(f["win_rate"]); g["b_win_rate_all"] = g["b"].map(f["win_rate"])
        g = pair_flags(g)
        g.to_csv(out / "pair_screens.csv", index=False)
        pair_core.last_links.to_csv(out / "pair_links.csv", index=False)
        rounds = int(sys.argv[sys.argv.index("--perm") + 1]) if "--perm" in sys.argv else PERMUTATIONS
        pl = placebo_pairs(b, t, rounds)
        pl.to_csv(out / "pair_placebo.csv", index=False)
        print(f"pairs {len(g)}; placebo rounds {len(pl)}")
    elif cmd == "market":
        out = Path(sys.argv[2])
        ts = pd.read_csv(out / "tender_screens.csv", low_memory=False)
        g = pd.read_csv(out / "pair_screens.csv") if (out / "pair_screens.csv").exists() else None
        links = pd.read_csv(out / "pair_links.csv") if (out / "pair_links.csv").exists() else None
        m = market_rollup(ts, g, links)
        m.to_csv(out / "market_screens.csv", index=False)
        print(f"markets {len(m)} (>= {MARKET_MIN_TENDERS} tenders)")
    elif cmd == "validate":
        out = Path(sys.argv[2]); lab = pd.read_csv(sys.argv[3], dtype=str)
        res = []
        for unit, fname, idcol, scores in (("market", "market_screens.csv", "market_id",
                                            ["composite", "price_score", "tie12_rate", "sync_rate", "flagged_pair_rate", "single_rate"]),):
            m = pd.read_csv(out / fname, dtype={idcol: str})
            l = lab[lab["unit_type"] == unit].merge(m, left_on="unit_id", right_on=idcol)
            for sc in scores:
                a, lo, hi = auc_ci(l.loc[l["label"] == "1", sc], l.loc[l["label"] == "0", sc])
                res.append(dict(unit=unit, score=sc, n_pos=int((l["label"] == "1").sum()),
                                n_neg=int((l["label"] == "0").sum()), auc=a, ci_lo=lo, ci_hi=hi))
        r = pd.DataFrame(res); r.to_csv(out / "validation.csv", index=False); print(r.to_string(index=False))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
