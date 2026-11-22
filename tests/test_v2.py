"""Unit tests for screens/v2.py on small synthetic bids (no real data in this public repo)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screens import labels, v2  # noqa: E402


def _t(rows):
    base = dict(yr=2023, status="x", item_key="desktop computers", buyer_id=1, ministry="M", organisation="O", state="S",
                n_received=3, n_tech=3, n_priced=3, award_value=1e6)
    return pd.DataFrame([{**base, **r} for r in rows])


def test_ra_join_links_subset_and_latest():
    t = _t([dict(tender_id=1, ref_type="B", item_id=7, published="2023-01-01 10:00", closing="2023-01-20 10:00"),
            dict(tender_id=2, ref_type="B", item_id=7, published="2023-01-05 10:00", closing="2023-02-01 10:00"),
            dict(tender_id=3, ref_type="R", item_id=7, published="2023-02-10 10:00", closing="2023-02-11 10:00"),
            dict(tender_id=4, ref_type="R", item_id=8, published="2023-02-10 10:00", closing="2023-02-11 10:00")])
    b = pd.DataFrame([dict(tender_id=tid, seller_id=s, seller_key=f"s{s}", tech=1, submitted_at="2023-01-10 10:00", price=100 + s,
                           rank_n=1, mse=False) for tid, ss in ((1, (1, 2, 3)), (2, (1, 2)), (3, (1, 2)), (4, (1, 2))) for s in ss])
    t, b = v2.prepare(t, b, "gem")
    link = v2.ra_join(t, b)
    assert link.set_index("ra_id")["b_id"].to_dict() == {3: 2}          # both B's contain {1,2}; tender 2 closed later


def test_effective_takes_ra_prices():
    t = _t([dict(tender_id=2, ref_type="B", item_id=7, published="2023-01-05", closing="2023-02-01"),
            dict(tender_id=3, ref_type="R", item_id=7, published="2023-02-10", closing="2023-02-11")])
    b = pd.DataFrame([dict(tender_id=2, seller_id=1, seller_key="a", tech=1, submitted_at=None, price=np.nan, rank_n=np.nan, mse=False),
                      dict(tender_id=2, seller_id=2, seller_key="b", tech=2, submitted_at=None, price=np.nan, rank_n=np.nan, mse=False),
                      dict(tender_id=3, seller_id=1, seller_key="a", tech=np.nan, submitted_at=None, price=90.0, rank_n=1, mse=False)])
    t, b = v2.prepare(t, b, "gem")
    te, be = v2.effective(t, b, pd.DataFrame({"ra_id": [3], "b_id": [2], "n_ra": [1]}))
    assert list(te["tender_id"]) == [2] and bool(te["has_ra"].iloc[0])
    row = be.set_index("seller_id").loc[1]
    assert row["price"] == 90.0 and row["tech_price"] == 1


def test_pair_count():
    T = np.array([[0.0, 5.0, 9.0, 30.0]])
    assert v2._pair_count(T, 10)[0] == 3


def test_sync2_flags_a_tight_cluster_only():
    rng = np.random.default_rng(1)
    rows, bids = [], []
    for i in range(60):
        rows.append(dict(tender_id=i, ref_type="B", item_id=7, published="2023-01-01 00:00", closing="2023-01-31 00:00"))
        for s in range(4):
            minute = rng.uniform(0, 28 * 24 * 60) if i else 1000 + s       # tender 0: four bids within 4 minutes
            bids.append(dict(tender_id=i, seller_id=s + 10 * i, seller_key=f"k{s}", tech=1,
                             submitted_at=pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=minute), price=100 + s, rank_n=s + 1, mse=False))
    t, b = v2.prepare(_t(rows), pd.DataFrame(bids), "gem")
    te, be = v2.effective(t, b, pd.DataFrame(columns=["ra_id", "b_id", "n_ra"]))
    ts = v2.tender_v2(te, be)
    r = ts.set_index("tender_id")
    assert r.loc[0, "sync2_flag"] == 1.0 and r.loc[0, "sync2_k"] == 6
    assert r["sync2_flag"].drop(0).mean() < 0.1


def test_closed_set_detects_a_ring():
    rows, bids = [], []
    for i in range(40):
        buyer = 1 if i < 20 else 2
        rows.append(dict(tender_id=i, ref_type="B", item_id=7, buyer_id=buyer, published="2023-01-01", closing="2023-01-31"))
        ss = [1, 2, 3][: 2 + i % 2] if buyer == 1 else [100 + i, 200 + i]           # buyer 1: the same three firms
        for s in ss:
            bids.append(dict(tender_id=i, seller_id=s, seller_key=f"k{s}", tech=1, submitted_at=None, price=100 + s, rank_n=1, mse=False))
    t, b = v2.prepare(_t(rows), pd.DataFrame(bids), "gepnic")
    te, be = v2.effective(t, b, pd.DataFrame(columns=["ra_id", "b_id", "n_ra"]))
    mk, tc = v2.closed_set(te, be, rounds=20)
    m = mk.set_index("m")
    assert m.loc["1|7|2023", "closed_share"] == 1.0 and m.loc["1|7|2023", "closed_flag"] == 1.0
    assert m.loc["2|7|2023", "closed_flag"] == 0.0


def test_labels_gem_toner_delhi():
    ts = pd.DataFrame({"tender_id": [1, 2, 3], "source": "gem", "yr": 2019, "item_key": ["toner cartridge", "toner", "paper"],
                       "state": ["Delhi", "Bihar", "Delhi"], "award_value": [3e6, 3e6, 3e6], "organisation": "x"})
    lab = labels.label(ts).set_index("tender_id")
    assert lab.loc[1, "label"] == 1 and lab.loc[2, "label"] == 0 and 3 not in lab.index
