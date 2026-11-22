"""A small made-up shard (random bids) for scrape-check and the check workflow; no real data."""
import numpy as np
import pandas as pd


def shard(n_tenders=120, seed=1):
    rng = np.random.default_rng(seed)
    t, b = [], []
    for i in range(n_tenders):
        pub = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
        t.append(dict(tender_id=i + 1, yr=2023, published=pub, closing=pub + pd.Timedelta(days=14), ref_type="B", status="x",
                      item_id=int(i % 3), item_key=f"class {i % 3}", buyer_id=int(i % 7), ministry="m", organisation=f"buyer {i % 7}",
                      state="s", n_received=0, n_tech=0, n_priced=0, award_value=float(rng.integers(1, 100)) * 1e4))
        sellers = rng.choice(40, size=int(rng.integers(1, 7)), replace=False)
        prices = np.sort(rng.uniform(90, 120, len(sellers)))
        for r, (s, p) in enumerate(zip(sellers, prices)):
            b.append(dict(tender_id=i + 1, seller_id=int(s), seller_key=f"seller{s}", tech=1 if rng.random() > 0.2 else 2,
                          submitted_at=pub + pd.Timedelta(minutes=float(rng.uniform(0, 13 * 1440))), price=float(p), rank_n=r + 1, mse=False))
    return pd.DataFrame(t), pd.DataFrame(b), None
