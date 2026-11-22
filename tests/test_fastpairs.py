"""fastpairs.pairs_of must return exactly what bidscreens_v1._pairs_of returns."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from screens import bidscreens_v1 as v1, fastpairs  # noqa: E402


def test_same_rows_as_v1():
    rng = np.random.default_rng(3)
    rows = []
    for t in range(300):
        n = int(rng.integers(1, 70))
        for s in rng.choice(500, size=n, replace=True):
            rows.append((t * 7 + 1, float(s)))
    df = pd.DataFrame(rows, columns=["tender_id", "seller_id"])
    a = v1._pairs_of(df).reset_index(drop=True)
    b = fastpairs.pairs_of(df)
    pd.testing.assert_frame_equal(a.astype(float), b.astype(float))
