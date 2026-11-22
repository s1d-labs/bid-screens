"""Bid screens, version 2 (definitions: preregistration-v2.md in the bet-3 research folder; frozen with this file's sha256).

Version 2 keeps the v1 functions where a definition did not change (tender price statistics, within-class percentiles, pair
core, label-permutation placebo) and adds: the GeM B-to-reverse-auction join, regulated-price classes out of price screens,
a synchronous-submission screen against a size- and timing-adjusted null, technical-disqualification screens, the cover
screen only with the firm control (C3), and a closed-bidder-set screen per buyer-market-year with its own placebo.
Every output is an indicator ("screen flags"), never a finding.
"""
import numpy as np
import pandas as pd

from . import bidscreens_v1 as v1

SEED = 20261122
RA_WINDOW_DAYS = 90            # a B bid closes at most 90 days before its reverse auction is published
REGULATED_RE = r"minimum wage|manpower|human resource outsourcing"   # PL2: wage-rule prices (GeM item class text)
SYNC_MIN, SYNC_EARLY_MIN = 10, 120          # as v1: two submissions <= 10 min apart, both >= 120 min before closing
SYNC_DRAWS, SYNC_MAX_N, SYNC_P = 200, 60, 0.01
POOL_MIN = 30                               # submissions needed in a class-year pool of timing positions (else the shard pool)
DQ_MIN_KNOWN, DQ2_MIN_SAME_MAKE = 3, 2
CORE_SHARE, CORE_MIN_TENDERS, CORE_MIN_FIRMS = 0.10, 3, 3
CLOSED_SHARE_MIN, CLOSED_P, CLOSED_ROUNDS = 0.5, 0.05, 50
C3_MIN_ORDERED, COVER_C3_MAX = 3, 0.5
PAIR_V2_ROUNDS = 200
MARKET_MIN_TENDERS = 10
FLAG_PCT = v1.FLAG_PCT


# ---------------------------------------------------------------- preparation
def prepare(t, b, source):
    """Types, test rows out, one row per seller per tender (lowest positive price, earliest submission), stage q/d/u."""
    t = t.copy(); b = b.copy()
    for c in ("published", "closing"):
        t[c] = v1._naive(t[c])
    b["submitted_at"] = v1._naive(b["submitted_at"])
    for c in ("price", "rank_n", "tech", "seller_id"):
        b[c] = v1._num(b[c])
    t = t[~t["item_key"].astype(str).str.contains("automationtestcategory", case=False, na=False)]
    b = b[b["tender_id"].isin(t["tender_id"]) & ~b["seller_key"].astype(str).str.fullmatch("TESTING", case=False, na=False)]
    if "stage" not in b:
        b["stage"] = np.where(b["tech"] == 1, "q", np.where(b["tech"] == 2, "d", "u"))
    if "make" not in b:
        b["make"] = np.nan
    b["price"] = b["price"].where(b["price"] > 0)
    b["_st"] = b["stage"].map({"q": 0, "d": 1}).fillna(2)
    b = (b.sort_values(["tender_id", "seller_id", "price"])
           .groupby(["tender_id", "seller_id"], as_index=False)
           .agg(seller_key=("seller_key", "first"), tech=("tech", "min"), submitted_at=("submitted_at", "min"),
                price=("price", "min"), rank_n=("rank_n", "min"), mse=("mse", "first"), make=("make", "first"), _st=("_st", "min")))
    b["stage"] = b["_st"].map({0: "q", 1: "d", 2: "u"}); b = b.drop(columns="_st")
    t["source"] = source
    t["regulated"] = t["item_key"].astype(str).str.contains(REGULATED_RE, case=False, regex=True) if source == "gem" else False
    return t, b


def ra_join(t, b):
    """GeM: link each reverse auction (ref_type R or RA) to its B bid. Rule (preregistration-v2 3.1): same item class, same
    ministry, the B closed 0-90 days before the RA was published, and every RA seller submitted in the B. Latest such B
    wins; a tie for latest = no link. A B with several RAs keeps the latest-published RA."""
    ra = t[t["ref_type"].astype(str) != "B"][["tender_id", "item_id", "ministry", "published"]]
    bb = t[t["ref_type"].astype(str) == "B"][["tender_id", "item_id", "ministry", "closing"]]
    empty = pd.DataFrame(columns=["ra_id", "b_id", "n_ra"])
    if ra.empty or bb.empty:
        return empty
    rs = b[b["tender_id"].isin(ra["tender_id"])][["tender_id", "seller_id"]].merge(ra, on="tender_id")
    rs = rs[rs["published"].notna() & rs["ministry"].notna() & (rs["ministry"].astype(str).str.strip() != "")]
    if rs.empty:
        return empty
    n_ra = rs.groupby("tender_id").size().rename("n_ra")
    rs["m"] = rs["published"].dt.year * 12 + rs["published"].dt.month
    rs = pd.concat([rs.assign(m=rs["m"] - k) for k in range(4)], ignore_index=True)
    bs = b[b["tender_id"].isin(bb["tender_id"])][["tender_id", "seller_id"]].merge(bb, on="tender_id")
    bs = bs[bs["closing"].notna()]
    bs["m"] = bs["closing"].dt.year * 12 + bs["closing"].dt.month
    x = rs.merge(bs, on=["item_id", "seller_id", "m"], suffixes=("_r", "_b"))
    x = x[(x["ministry_r"] == x["ministry_b"]) & (x["closing"] <= x["published"])
          & (x["closing"] >= x["published"] - pd.Timedelta(days=RA_WINDOW_DAYS))]
    if x.empty:
        return empty
    ov = x.groupby(["tender_id_r", "tender_id_b"]).agg(ov=("seller_id", "nunique"), closing=("closing", "first"),
                                                       published=("published", "first")).reset_index()
    ov = ov.merge(n_ra, left_on="tender_id_r", right_index=True)
    ov = ov[ov["ov"] == ov["n_ra"]]
    last = ov.groupby("tender_id_r")["closing"].transform("max")
    ov = ov[ov["closing"] == last]
    ov = ov[ov.groupby("tender_id_r")["tender_id_b"].transform("size") == 1]             # a tie for latest: no link
    ov = ov[ov["published"] == ov.groupby("tender_id_b")["published"].transform("max")]    # the final RA of a B
    ov = ov.drop_duplicates("tender_id_b", keep="last")
    return ov.rename(columns={"tender_id_r": "ra_id", "tender_id_b": "b_id"})[["ra_id", "b_id", "n_ra"]]


def effective(t, b, link):
    """B tenders only; a linked B takes its RA's prices and ranks (RA sellers count as technically qualified)."""
    te = t[t["ref_type"].astype(str) == "B"].copy()
    te["has_ra"] = te["tender_id"].isin(link["b_id"])
    be = b[b["tender_id"].isin(te["tender_id"])].copy()
    if len(link):
        r = b[b["tender_id"].isin(link["ra_id"])][["tender_id", "seller_id", "price", "rank_n"]]
        r = r.merge(link[["ra_id", "b_id"]], left_on="tender_id", right_on="ra_id")[["b_id", "seller_id", "price", "rank_n"]]
        r = r.rename(columns={"b_id": "tender_id", "price": "ra_price", "rank_n": "ra_rank"})
        be = be.merge(r, on=["tender_id", "seller_id"], how="left")
        lk = be["tender_id"].isin(link["b_id"])
        be.loc[lk, "price"] = be.loc[lk, "ra_price"]
        be.loc[lk, "rank_n"] = be.loc[lk, "ra_rank"]
        be["tech_price"] = np.where(lk, np.where(be["ra_price"] > 0, 1, np.nan), be["tech"])
        be = be.drop(columns=["ra_price", "ra_rank"])
    else:
        be["tech_price"] = be["tech"]
    return te, be


# ---------------------------------------------------------------- tender screens
def tender_v2(te, be):
    """v1 price statistics on effective prices (regulated classes masked), within-class percentiles, then v2 screens."""
    bp = be.copy()
    bp["tech"] = bp["tech_price"]
    reg = te.set_index("tender_id")["regulated"]
    bp.loc[bp["tender_id"].map(reg).fillna(False).astype(bool), "price"] = np.nan
    ts = v1.within_class_percentiles(v1.tender_screens(te, bp))
    ts = ts.rename(columns={"price_screen_score": "price_rank", "sync_pairs": "s7_pairs"})
    ts = ts.merge(te[["tender_id", "has_ra", "regulated", "source", "published", "closing"]
                     + [c for c in ("ref", "title", "org_id", "category") if c in te]], on="tender_id", how="left")
    ts = ts.merge(sync2(ts, be), on="tender_id", how="left")
    ts = ts.merge(dq(ts, be), on="tender_id", how="left")
    return ts


def _pair_count(T, d):
    """T: (..., n) sorted times in minutes -> number of pairs <= d apart, per leading index."""
    n = T.shape[-1]
    k = np.zeros(T.shape[:-1], dtype=np.int64)
    for s in range(1, n):
        k += (T[..., s:] - T[..., :-s] <= d).sum(-1)
    return k


def sync2(ts, be, rng=None):
    """Synchronous submission against a size- and timing-adjusted null (preregistration-v2 3.3). For a tender with n early
    submissions (>= 120 min before closing), K = pairs <= 10 min apart; the null draws n timing positions
    u = (t - published) / (closing - 120 min - published) from the same class-year's early submissions, 200 times."""
    rng = rng or np.random.default_rng(SEED)
    x = be[["tender_id", "submitted_at"]].merge(ts[["tender_id", "published", "closing", "norm_group"]], on="tender_id")
    x["end"] = x["closing"] - pd.Timedelta(minutes=SYNC_EARLY_MIN)
    x = x[x["submitted_at"].notna() & x["published"].notna() & x["closing"].notna() & (x["end"] > x["published"])
          & (x["submitted_at"] >= x["published"]) & (x["submitted_at"] <= x["end"])].copy()
    out = pd.DataFrame({"tender_id": ts["tender_id"]})
    if x.empty:
        return out.assign(n_early=0, sync2_k=np.nan, sync2_p=np.nan, sync2_score=np.nan, sync2_flag=np.nan, sync2_pct=np.nan)
    x["W"] = (x["end"] - x["published"]).dt.total_seconds() / 60.0
    x["tm"] = (x["submitted_at"] - x["published"]).dt.total_seconds() / 60.0
    x["u"] = (x["tm"] / x["W"]).clip(0, 1)
    shard_pool = x["u"].values
    pools = {g: v.values for g, v in x.groupby("norm_group")["u"]}
    per = x.groupby("tender_id").agg(n_early=("tm", "size"), W=("W", "first"), g=("norm_group", "first"))
    kobs = x.sort_values(["tender_id", "tm"]).groupby("tender_id")["tm"].apply(
        lambda s: int(_pair_count(s.values[None, :], SYNC_MIN)[0]))
    per["sync2_k"] = kobs
    per["sync2_p"] = np.nan
    ok = (per["n_early"] >= 2) & (per["n_early"] <= SYNC_MAX_N)
    per.loc[ok & (per["sync2_k"] == 0), "sync2_p"] = 1.0
    todo = per[ok & (per["sync2_k"] > 0)]
    for (g, n), grp in todo.groupby([todo["g"].fillna("__shard__"), "n_early"]):
        pool = pools.get(g) if g != "__shard__" else None
        if pool is None or len(pool) < POOL_MIN:
            pool = shard_pool
        m = len(grp)
        step = max(1, int(2e7 // (SYNC_DRAWS * n)))
        for i in range(0, m, step):
            sub = grp.iloc[i:i + step]
            U = rng.choice(pool, size=(len(sub), SYNC_DRAWS, int(n)), replace=True)
            T = np.sort(U * sub["W"].values[:, None, None], axis=-1)
            K = _pair_count(T, SYNC_MIN)
            ge = (K >= sub["sync2_k"].values[:, None]).sum(1)
            per.loc[sub.index, "sync2_p"] = (1 + ge) / (SYNC_DRAWS + 1)
    per["sync2_score"] = -np.log10(per["sync2_p"])
    per["sync2_flag"] = (per["sync2_p"] <= SYNC_P).astype(float).where(per["sync2_p"].notna())
    out = out.merge(per[["n_early", "sync2_k", "sync2_p", "sync2_score", "sync2_flag"]], left_on="tender_id", right_index=True, how="left")
    out["n_early"] = out["n_early"].fillna(0)
    g = ts.set_index("tender_id")["norm_group"].reindex(out["tender_id"]).values
    out["sync2_pct"] = out["sync2_score"].groupby(g).rank(pct=True, method="average").values
    return out


def dq(ts, be):
    """DQ1: share of sellers rejected at the technical stage (stage d) among those with a known stage, >= 3 known;
    percentile within class-year, flag = top 5 % and share > 0. DQ2 (makes known, GeM): >= 2 sellers offering the
    winner's make were rejected at the technical stage while the winner (effective L1) passed."""
    s = be.assign(q=(be["stage"] == "q").astype(int), d=(be["stage"] == "d").astype(int))
    a = s.groupby("tender_id").agg(n_q=("q", "sum"), n_d=("d", "sum"))
    a["known"] = a["n_q"] + a["n_d"]
    a["dq_share"] = (a["n_d"] / a["known"]).where(a["known"] >= DQ_MIN_KNOWN)
    g = ts.set_index("tender_id")["norm_group"].reindex(a.index)
    a["dq_pct"] = a["dq_share"].groupby(g.values).rank(pct=True, method="average").values
    a["dq_flag"] = ((a["dq_pct"] >= 1 - FLAG_PCT) & (a["dq_share"] > 0)).astype(float).where(a["dq_pct"].notna())
    w = s[(s["rank_n"] == 1) & (s["stage"] == "q")]
    w = w[w.groupby("tender_id")["seller_id"].transform("size") == 1][["tender_id", "seller_id", "make"]]
    w = w[w["make"].notna() & (w["make"].astype(str) != "")]
    if len(w):
        y = s.merge(w.rename(columns={"seller_id": "w_id", "make": "w_make"}), on="tender_id")
        y = y[(y["seller_id"] != y["w_id"]) & (y["make"] == y["w_make"]) & (y["stage"] == "d")]
        n_same = y.groupby("tender_id").size()
        a["dq2_n"] = np.where(a.index.isin(w["tender_id"]), n_same.reindex(a.index).fillna(0).values, np.nan)
        a["dq2"] = (a["dq2_n"] >= DQ2_MIN_SAME_MAKE).astype(float).where(a["dq2_n"].notna())
    else:
        a["dq2_n"] = np.nan; a["dq2"] = np.nan
    return a[["n_q", "n_d", "dq_share", "dq_pct", "dq_flag", "dq2_n", "dq2"]].reset_index()


# ---------------------------------------------------------------- pairs
def pairs_v2(te, be, dirs=None, rounds=PAIR_V2_ROUNDS):
    """v1 pair core on effective prices and ranks; cover (P2) only with the firm control C3: the always-higher firm is higher
    in < 50 % of its ordered joint tenders with its other rivals (>= 3 ordered, rivals with >= 5 joint tenders)."""
    bp = be[be["tech"].isin([1, 2, 3]) | be["tech"].isna()].copy()
    g = v1.pair_core(bp, te)
    if g is None or len(g) == 0:
        return pd.DataFrame(), pd.DataFrame(columns=["tender_id", "a", "b"]), pd.DataFrame()
    links = v1.pair_core.last_links
    f = v1.firm_rates(bp)
    g = v1.shared_directors(g, dirs, f)
    g["a_key"] = g["a"].map(f["seller_key"]); g["b_key"] = g["b"].map(f["seller_key"])
    g["a_win_rate_all"] = g["a"].map(f["win_rate"]); g["b_win_rate_all"] = g["b"].map(f["win_rate"])
    g = v1.pair_flags(g)
    hi_is_b = g["a_lower"] >= g["b_lower"]                         # the higher (losing) firm of the pair
    g["hi"] = np.where(hi_is_b, g["b"], g["a"])
    g["hi_cnt"] = np.where(hi_is_b, g["a_lower"], g["b_lower"])    # tenders where hi was above its rival
    tot_hi = pd.concat([pd.Series(g["a_lower"].values, index=g["b"].values), pd.Series(g["b_lower"].values, index=g["a"].values)])
    tot_hi = tot_hi.groupby(level=0).sum()
    tot_ord = pd.concat([pd.Series(g["n_both_priced"].values, index=g["a"].values),
                         pd.Series(g["n_both_priced"].values, index=g["b"].values)]).groupby(level=0).sum()
    den = g["hi"].map(tot_ord) - g["n_both_priced"]
    g["c3_hi_above_other"] = ((g["hi"].map(tot_hi) - g["hi_cnt"]) / den).where(den >= C3_MIN_ORDERED)
    g["flag_cover_c3"] = g["flag_cover"] & (g["c3_hi_above_other"] < COVER_C3_MAX)
    g["flag_any_v2"] = g[["flag_rotation", "flag_sync", "flag_director", "flag_cover_c3"]].any(axis=1)
    pl = v1.placebo_pairs(bp, te, rounds) if rounds else pd.DataFrame()
    return g, links, pl


# ---------------------------------------------------------------- closed bidder set
def _closed(rows):
    """rows: tender_id, seller_id, m (market-year) of tenders with >= 2 bidders -> per market: n_t, n_core, closed_share;
    per tender: closed (all bidders in the market's core set)."""
    nt = rows.groupby("m")["tender_id"].nunique()
    cnt = rows.groupby(["m", "seller_id"]).size().rename("c").reset_index()
    thr = np.maximum(CORE_MIN_TENDERS, np.ceil(CORE_SHARE * cnt["m"].map(nt)))
    cnt["core"] = cnt["c"] >= thr
    r = rows.merge(cnt[["m", "seller_id", "core"]], on=["m", "seller_id"])
    tc = r.groupby(["m", "tender_id"])["core"].all().rename("closed").reset_index()
    mk = tc.groupby("m")["closed"].mean().rename("closed_share").to_frame()
    mk["n_t"] = nt
    mk["n_core"] = cnt.groupby("m")["core"].sum()
    return mk, tc


def closed_set(te, be, rounds=CLOSED_ROUNDS, rng=None):
    """Closed-bidder-set screen per buyer-market-year (preregistration-v2 3.6), with a label-permutation placebo that shuffles
    sellers within item class x year (GeM: x buyer state), keeping tender sizes."""
    rng = rng or np.random.default_rng(SEED + 1)
    x = be[["tender_id", "seller_id"]].merge(te[["tender_id", "buyer_id", "item_id", "yr", "state"]], on="tender_id")
    nb = x.groupby("tender_id")["seller_id"].transform("nunique")
    x = x[nb >= 2].copy()
    x["state"] = x["state"].fillna("").astype(str)               # a missing buyer state is its own stratum
    if x.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["tender_id", "closed_t"])
    x["m"] = x["buyer_id"].astype(str) + "|" + x["item_id"].astype(str) + "|" + x["yr"].astype(str)
    strata = ["item_id", "yr", "state"] if te["source"].iloc[0] == "gem" else ["item_id", "yr"]
    mk, tc = _closed(x[["tender_id", "seller_id", "m"]])
    elig = mk.index[mk["n_t"] >= MARKET_MIN_TENDERS]
    sims = []
    xe = x[x["m"].isin(elig)].copy()
    for _ in range(rounds if len(elig) else 0):
        y = xe.copy()
        y["seller_id"] = y.groupby(strata)["seller_id"].transform(lambda s: rng.permutation(s.values))
        y = y.drop_duplicates(["tender_id", "seller_id"])
        ms, _ = _closed(y[["tender_id", "seller_id", "m"]])
        sims.append(ms["closed_share"].reindex(elig))
    if sims:
        S = pd.concat(sims, axis=1)
        obs = mk.loc[elig, "closed_share"]
        mk.loc[elig, "closed_p"] = (1 + S.ge(obs, axis=0).sum(1)) / (len(sims) + 1)
        mk.loc[elig, "closed_placebo_mean"] = S.mean(1)
        mk.loc[elig, "closed_excess"] = obs - S.mean(1)
    for c in ("closed_p", "closed_placebo_mean", "closed_excess"):
        if c not in mk:
            mk[c] = np.nan
    mk["closed_flag"] = ((mk["n_t"] >= MARKET_MIN_TENDERS) & (mk["n_core"] >= CORE_MIN_FIRMS)
                         & (mk["closed_share"] >= CLOSED_SHARE_MIN) & (mk["closed_p"] <= CLOSED_P)).astype(float)
    mk.loc[mk["n_t"] < MARKET_MIN_TENDERS, "closed_flag"] = np.nan
    tc = tc.merge(mk[["closed_flag"]], left_on="m", right_index=True)
    tc["closed_t"] = (tc["closed"] & (tc["closed_flag"] == 1)).astype(float).where(tc["closed_flag"].notna())
    return mk.reset_index(), tc[["tender_id", "closed_t"]]


# ---------------------------------------------------------------- composites and market rollup
TENDER_PARTS = ["price_rank", "sync2_pct", "dq_pct", "closed_t", "pair_v2_t"]
MARKET_PARTS = ["price_rank", "sync2_rate", "dq_rate", "closed_excess", "pair_v2_rate"]


def finish(ts, g, links, cs_t):
    """Tender-level pair and closed indicators and the tender composite tender_v2 (mean of the available parts)."""
    if cs_t is not None and len(cs_t):
        ts = ts.merge(cs_t, on="tender_id", how="left")
    else:
        ts["closed_t"] = np.nan
    if g is not None and len(g):
        fl = links.merge(g.loc[g["flag_any_v2"], ["a", "b"]], on=["a", "b"])
        ts["pair_v2_t"] = ts["tender_id"].isin(fl["tender_id"]).astype(float)
    else:
        ts["pair_v2_t"] = 0.0
    ts.loc[ts["n_submitted"] < 2, "pair_v2_t"] = np.nan
    for c in TENDER_PARTS:                       # A3: a part a shard cannot compute is NaN (e.g. no submission times)
        if c not in ts:
            ts[c] = np.nan
    ts["tender_v2"] = ts[TENDER_PARTS].mean(axis=1, skipna=True)
    return ts


def market_v2(ts, mk_closed):
    ts = ts.copy()
    for c in ("tie12", "single_bid", "sync2_flag", "dq_flag", "dq2", "price_rank", "pair_v2_t", "tender_v2"):
        if c not in ts:                                            # a shard without priced tenders has no tie12 (A1)
            ts[c] = np.nan
    ts["m"] = ts["buyer_id"].astype(str) + "|" + ts["item_id"].astype(str) + "|" + ts["yr"].astype(str)
    ts["price_rank_nr"] = ts["price_rank"].where(~ts["regulated"].astype(bool))
    m = ts.groupby("m").agg(source=("source", "first"), buyer_id=("buyer_id", "first"), item_id=("item_id", "first"),
                            yr=("yr", "first"), organisation=("organisation", "first"), item_key=("item_key", "first"),
                            state=("state", "first"), regulated=("regulated", "first"), n_tenders=("tender_id", "size"),
                            n_price=("price_rank_nr", "count"), price_rank=("price_rank_nr", "mean"),
                            single_rate=("single_bid", "mean"), tie_rate=("tie12", "mean"), sync2_rate=("sync2_flag", "mean"),
                            dq_rate=("dq_flag", "mean"), dq2_rate=("dq2", "mean"), pair_v2_rate=("pair_v2_t", "mean"),
                            has_ra_rate=("has_ra", "mean"), value=("award_value", "sum"),
                            tender_v2_mean=("tender_v2", "mean")).reset_index()
    m.loc[m["n_price"] < 3, "price_rank"] = np.nan
    if mk_closed is not None and len(mk_closed):
        m = m.merge(mk_closed[["m", "n_core", "closed_share", "closed_p", "closed_excess", "closed_flag"]], on="m", how="left")
    else:
        for c in ("n_core", "closed_share", "closed_p", "closed_excess", "closed_flag"):
            m[c] = np.nan
    return m


def national_composite(m):
    """Across all market-years of one source with >= 10 tenders: mean of the parts' percentile ranks (missing parts skipped)."""
    m = m[m["n_tenders"] >= MARKET_MIN_TENDERS].copy()
    for c in MARKET_PARTS:
        m[c + "_r"] = m.groupby("source")[c].rank(pct=True)
    m["composite_v2"] = m[[c + "_r" for c in MARKET_PARTS]].mean(axis=1, skipna=True)
    return m
