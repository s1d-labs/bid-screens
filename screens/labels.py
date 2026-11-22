"""Held-out validation cases (preregistration-v2 section 6), fixed before any v2 value was computed.

Keys come from the CCI order text only (bet-3 ground-truth.md): portal, buyer words, product words, years, value floor.
Positives = tenders matching the case rule; controls = tenders of the same portal (GeM: same product and value floor), same
years and product family, that are not positives. Labels are written per shard; the rank task pools them. A case label says
"this tender falls in the market a CCI order describes"; it does not say any tender or firm in it was rigged.
"""
import numpy as np
import pandas as pd

CASES = {
    # CCI Case 34/2015 (order 14.09.2017): Western Coalfields, sand and coal transport, NITs of 2013-14 and 2014-15
    "wcl": dict(source="gepnic", portal="gepnic-coalindiatenders", years=(2013, 2015), field="title",
                pos_org=r"western coalfield", pos=r"transport", ctrl=r"transport", primary=True),
    # CCI Ref 03/2013 (05.10.2017): Delhi Jal Board, poly aluminium chloride, NIT 01(12-13), 2006-12
    "djb": dict(source="gepnic", portal="gepnic-govtprocurement-delhi", years=(2011, 2013), field="title",
                pos_org=r"delhi jal board", pos=r"poly\s*alumini?um\s*chloride|\bpac\b",
                ctrl=r"chlorine|\balum\b|alumini?um\s*sulphate|bleaching|\blime\b|polyelectrolyte|caustic|chemical", primary=True),
    # CCI Ref 01/2021 (02.01.2026): DGOS / MGO (Army), woollen underpants, tenders of 11.07.2019 and 12.11.2020
    "dgos": dict(source="gepnic", portal="gepnic-defproc", years=(2019, 2020), field="title", pos_org=r"",
                 pos=r"under\s*-?\s*pant", ctrl=r"\bvests?\b|socks|jersey|trouser|shirt|drawers|woollen|garment|cap\s*comforter|gloves",
                 primary=True),
    # CCI SM 08/2020 (13.07.2026): GeM buyers in Delhi, toner and ink, 29 tenders > Rs 20 lakh, 2017-20 (no IDs public)
    "hp_toner": dict(source="gem", years=(2018, 2020), field="item_key", pos=r"toner|cartridge", ctrl=r"toner|cartridge",
                     pos_state=r"delhi", min_value=2_000_000, primary=True),
    # CCI Ref 05/2013 (19.01.2017): DS&D Haryana, cement, 2012-13 (3 tenders: secondary, not pooled)
    "hry_cement": dict(source="gepnic", portal="gepnic-etenders-hry", years=(2012, 2013), field="title",
                       pos_org=r"suppl(y|ies)\s*(and|&)\s*disposal", pos=r"\bcement\b", ctrl=r"\bcement\b", primary=False),
}


def label(ts):
    """ts: v2 tender table of one shard -> rows (tender_id, case, label) for tenders that are a positive or a control."""
    out = []
    src = ts["source"].iloc[0] if len(ts) else ""
    for case, c in CASES.items():
        if c["source"] != src:
            continue
        x = ts
        if src == "gepnic":
            x = x[x["state"] == c["portal"]]
        x = x[x["yr"].between(*c["years"])]
        if x.empty:
            continue
        text = x[c["field"]].astype(str)
        if src == "gepnic":
            buyer = x["organisation"].astype(str)
            pos = text.str.contains(c["pos"], case=False, regex=True) & (
                buyer.str.contains(c["pos_org"], case=False, regex=True) if c["pos_org"] else True)
            ctrl = text.str.contains(c["ctrl"], case=False, regex=True) & ~pos
            if c["pos_org"] and c["pos"] == c["ctrl"]:          # same product: controls are the other buyers
                ctrl &= ~buyer.str.contains(c["pos_org"], case=False, regex=True)
        else:
            val = pd.to_numeric(x["award_value"], errors="coerce")
            inval = val >= c["min_value"]
            prod = text.str.contains(c["pos"], case=False, regex=True) & inval
            delhi = x["state"].astype(str).str.contains(c["pos_state"], case=False, regex=True)
            pos, ctrl = prod & delhi, prod & ~delhi
        lab = pd.Series(np.where(pos, 1, np.where(ctrl, 0, -1)), index=x.index)
        y = x.loc[lab >= 0, ["tender_id"]].assign(case=case, label=lab[lab >= 0].values, primary=c["primary"])
        out.append(y)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["tender_id", "case", "label", "primary"])
