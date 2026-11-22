"""A vectorised drop-in for bidscreens_v1._pairs_of (speed only: the same rows, the same order).

v1 builds one small DataFrame per tender, which made 200 placebo rounds cost about 0.3-1 ms per tender per round. This
function returns exactly what v1._pairs_of returns (every unordered pair of distinct sellers, a < b, of tenders with 2-60
distinct sellers, sorted by tender then a then b); tests/test_fastpairs.py checks the equality. The v1 file stays unchanged.
"""
import numpy as np
import pandas as pd


def pairs_of(df):
    s = df[["tender_id", "seller_id"]].drop_duplicates()
    n = s.groupby("tender_id")["seller_id"].transform("size")
    s = s[(n >= 2) & (n <= 60)]
    if s.empty:
        return pd.DataFrame(columns=["tender_id", "a", "b"])
    x = s.merge(s, on="tender_id", suffixes=("_a", "_b"))
    x = x[x["seller_id_a"] < x["seller_id_b"]]
    x = x.rename(columns={"seller_id_a": "a", "seller_id_b": "b"}).sort_values(["tender_id", "a", "b"], kind="mergesort")
    return x[["tender_id", "a", "b"]].reset_index(drop=True)


def install(v1):
    v1._pairs_of = pairs_of
