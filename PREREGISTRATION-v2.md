# Pre-registration, version 2: bid screens on Indian public procurement (bet 3, national run)

Status: **FROZEN** at the time and code hashes in section 10, before any version-2 screen value was computed. Version 1
(`preregistration.md`, frozen 05:35 IST, amendments A1-A4) stays in force for the v1 screens; the national run computes
v1 and v2 side by side. What the pilot showed (`pilot/results.md` section 7) motivated each change below; the pilot cases
(CCI SM 07/2020, SM 01/2020, SM 03/2021) are therefore **not** used to judge v2. v2 is judged only on the held-out cases in
section 6. Any later change goes into section 11 with a date and a reason.

Code: repository `s1d-labs/bid-screens` (scrape-hub family `bid-screens`). `screens/bidscreens_v1.py` is the v1 file byte for
byte (sha256 `5db34449…`); `screens/v2.py` holds every v2 definition; `screens/labels.py` the validation cases;
`run_shard.py` the runner task; `rank.py` the national step.

## 1. Questions (unchanged from v1, plus Q4)
- Q1 base rates of each screen nationally, by source and year. Q2 validation against CCI markets. Q3 lead yield.
- Q4: do the v2 screens rank held-out CCI tenders above matched control tenders better than chance, and better than v1?

## 2. Data
- GeM: `gem_lab` `lab.bid_core`, `lab.bidder_core`, `lab.bid` (bid number, published and closing **timestamps**), `lab.dim_*`,
  `lab.hr_bidder` (the make each seller offered), directors as in v1. Years 2018-2026 (to 10 Sep 2026). Test bids out.
- GePNIC: `scrape_hub.records` of 43 portal projects, every tender with a bid list, one portal-year at a time; amounts from
  `doc-ocr-gepnic-awards` where read. Raw bid status kept; `stage` = q (passed the technical stage: any Admitted/Accepted
  status, or Rejected/Not Admitted at Finance or AOC), d (rejected at fee / pre-qualification / technical stage), u (other).
- Shards: GeM = whole item classes over all years (a class with > 120,000 bid rows is cut by year; then each year shard also
  holds the reverse auctions published up to 31 March of the next year). GePNIC = portal x tender-ID year, cut into whole
  org_ids (an item class there is org_id x category). Pairs and placebos are counted inside a shard.
- Data fix for v1 too: GeM closing time is now the closing timestamp (the pilot used the closing date, i.e. midnight).

## 3. Version-2 screens (definitions = `screens/v2.py`)
3.1 **B-to-reverse-auction join (GeM).** A reverse auction (ref_type R or RA) links to the B bid with the same item class and
ministry, closing 0-90 days before the RA was published, in which **every** RA seller submitted. The latest such B wins; a
tie = no link. A B with several RAs keeps the latest-published one. A linked B uses the RA prices and ranks (RA sellers count
as technically qualified); RA rows are not tenders of their own. Unlinked RAs are dropped and counted.

3.2 **Regulated-price classes out of price screens.** GeM item classes matching `minimum wage|manpower|human resource
outsourcing` get no price statistics in v2 (PL2: wage rules fix their prices). They keep every other screen.

3.3 **Price screens.** v1 S2-S6 on effective prices, within-class-year percentiles (v1 rule). `price_rank` = mean of the
CV, RD, kurtosis and PD percentiles (needs all four). It **ranks**; it is not a flag (pilot: it never reached the top 5 %).

3.4 **SYNC2: synchronous submission against a size- and timing-adjusted null.** Early submissions = those >= 120 min before
closing and after publication. K = pairs <= 10 min apart among them. Null: draw n timing positions u = (t - published) /
(closing - 120 min - published) with replacement from all early submissions of the same class-year (norm_group; the shard
pool when the group has < 30), 200 draws, seed 20261122. p = (1 + #{K_sim >= K}) / 201 (p = 1 when K = 0). Only 2 <= n <= 60.
`sync2_flag` = p <= 0.01. `sync2_pct` = percentile of -log10 p within class-year.

3.5 **DQ: technical disqualification.** DQ1 `dq_share` = sellers with stage d / sellers with a known stage (>= 3 known);
flag = top 5 % of its class-year and share > 0. DQ2 (GeM, makes known) = >= 2 sellers offering the winner's make were
rejected at the technical stage while the winner (unique effective L1) passed (the withheld-authorisation pattern).

3.6 **CLOSED: closed bidder set (market-year).** Market-year m = buyer x item class x year, tenders with >= 2 bidders,
>= 10 such tenders. Core set C_m = sellers in >= max(3, 10 % of m's tenders). closed(t) = every bidder of t is in C_m.
`closed_share` = mean closed(t). Placebo: sellers shuffled within item class x year (GeM: x buyer state), tender sizes kept,
50 rounds, seed 20261123; `closed_p` = (1 + #{placebo >= observed}) / 51; `closed_excess` = observed - placebo mean.
`closed_flag` = closed_share >= 0.5 and closed_p <= 0.05 and |C_m| >= 3. Tender indicator `closed_t` = closed(t) in a
flagged market.

3.7 **Pairs.** v1 P1 rotation, P3 synchronous, P4 shared director unchanged, on effective prices and ranks. **P2 cover only
with the firm control C3:** the always-higher firm h must be higher in < 50 % of its ordered joint tenders with its other
rivals (rivals with >= 5 joint tenders; >= 3 ordered tenders) - "h is cheap against others but always above this one".
`flag_any_v2` = P1 or P3 or P4 or P2+C3. Placebo PL1 for v2 pairs: 200 rounds (v1 procedure).

3.8 **Composites.** Tender: `tender_v2` = mean of the available parts among price_rank, sync2_pct, dq percentile, closed_t,
pair_v2_t (tender has a v2-flagged pair). Market-year: `composite_v2` = mean of the national percentile ranks (within source,
market-years with >= 10 tenders) of price_rank (non-regulated, >= 3 priced tenders), sync2_rate, dq_rate, closed_excess,
pair_v2_rate. S1 single-bid rate is reported beside it, not inside it.

## 4. v1 in the national run
v1 runs unchanged on every shard (tender screens, pair screens with PL1 200 rounds as pre-registered, market rollup). Speed
only: `_pairs_of` is replaced at run time by a vectorised function that returns the same rows (`screens/fastpairs.py`,
equality test `tests/test_fastpairs.py`); the v1 file is byte-identical. v1 market composites are re-ranked nationally in
`rank.py` (v1 ranked within one scope).

## 5. Placebos and controls
PL1 (labels shuffled) for pairs (v1 and v2) and for CLOSED; PL2 regulated classes: v1 price flags reported there as the
lawful false-positive rate (v2 masks them); the SYNC2 null is itself a timing placebo. C3 is part of P2 (3.7).

## 6. Held-out validation (fixed now; `screens/labels.py`)
Keys from the order text only (`ground-truth.md`). Positive tenders and matched control tenders:

| Case | Order | Positives | Controls |
|---|---|---|---|
| `wcl` | 34/2015, 14.09.2017 | `gepnic-coalindiatenders`, tender-ID years 2013-2015, buyer `western coalfield`, title `transport` | same portal and years, title `transport`, other buyers |
| `djb` | Ref 03/2013, 05.10.2017 | `gepnic-govtprocurement-delhi`, 2011-2013, buyer `delhi jal board`, title poly aluminium chloride / PAC | same portal and years, other water-treatment chemical titles (chlorine, alum, aluminium sulphate, bleaching, lime, polyelectrolyte, caustic, chemical), any buyer |
| `dgos` | Ref 01/2021, 02.01.2026 | `gepnic-defproc`, 2019-2020, title `underpant` | same portal and years, other clothing titles (vest, socks, jersey, trouser, shirt, drawers, woollen, garment, cap comforter, gloves) |
| `hp_toner` | SM 08/2020, 13.07.2026 | GeM, 2018-2020, item class `toner|cartridge`, buyer state Delhi, value >= Rs 20 lakh | same class words, years and value floor, buyer state not Delhi |
| `hry_cement` (secondary, not pooled) | Ref 05/2013, 19.01.2017 | `gepnic-etenders-hry`, 2012-2013, buyer Supplies and Disposals, title `cement` | same portal, years and title, other buyers |

Label noise is expected and accepted: a case's positive set contains tenders that the order may not cover (no tender IDs
are public for most cases); this biases the AUC towards 0.5, not away from it.

**Primary metric (Q4):** pooled within-case AUC of `tender_v2`, positives vs controls of the same case, over the four
primary cases (ties count 1/2). 95 % CI: stratified bootstrap (resample positives and controls within each case), 2,000
rounds, seed 20261122. **Success = lower CI bound > 0.5.** Reported beside it: per-case AUC with CI; each v2 part alone; v1
comparators (`price_screen_score`, v1 S7 count) on the same tenders. With < 5 positives in a case its AUC is shown but
called "too few". **Secondary (market level):** market-years holding >= 1 positive vs those holding >= 1 control, >= 10
tenders: AUC of composite_v2 and closed_excess when >= 5 positive market-years, else each positive's percentile.

## 7. Lead filter and wording (unchanged policy, v1 section 7)
Lead candidates for the Overseer list: market-years with composite_v2 in the top 1 % of their source, >= 10 tenders, not a
regulated class, ranked by composite_v2, one per buyer; plus v2 pairs that flag on >= 2 pair screens, or on P2+C3 or P1,
with the shard's observed count above its placebo 95th percentile. Every one is an **indicator** ("screen flags"); the
lawful-explanation checks of v1 section 7 come before anything else. Internal only; never published.

## 8. Compute
Runner tasks on the scrape-hub pool (family `bid-screens`), <= 40 concurrent jobs (80 worker processes), one shard per
process, inputs and outputs in the family owner's bucket under the project prefix. Exports on dev-vps-3 at nice 15;
`scrape_hub.records` only per project and key range, 21:00-09:00 IST, one portal-year at a time.

## 9. Not tested
Same as v1 section 9; also: no dated directors (P4 stays current directors), no bid-document metadata.

## 10. Freeze record
- Frozen: **2026-11-22 06:48 IST**, before any v2 value of the national run was computed and before any held-out label was
  joined to a score. Before the freeze, runtime smoke tests ran the code on three real shards (GePNIC coalindiatenders 2014
  and defproc 2019, GeM desktop computers 2023): only runtimes and shard-wide counts (tenders, pairs, RA links, eligible
  counts) were printed; no flag rate of a labelled case, no labelled tender and no validation number was looked at.
- Repository `s1d-labs/bid-screens` commit `568907c7c63ed214dbd6f1f94647eb0a6f53d38a`.
- sha256: `screens/bidscreens_v1.py` 5db34449dc250a9d75ae42c3f8467982f185fb26729e972e11e623fa0cf14155 (= v1 freeze, A3);
  `screens/v2.py` 8146c6e39cf5a6c7c77371e4af2e6be0402fb4c3f5adeadd7d77369c945e0da8;
  `screens/labels.py` 1cc98eb6e69b0817dfb4dcec20924ae2e37347b95e113b9debec269ad081fe56;
  `screens/fastpairs.py` 34c82164c75eedb363d70fae204870e5b1f356cec160b95c8afa175cc21a5fdb;
  `run_shard.py` 4712944e5e07681633f35f24b504f3d81da33bddee9b395f67842e9b0f6e52bf;
  `rank.py` ebffd641b74f7830a2d71fc3044bef21e8c7ace621d7b7240aff038a98c5d6c5.
- Pinned code hash in every queue item (`run_shard.code_sha`, over v1, v2, labels, fastpairs, run_shard):
  `1ca7d9c802f6130c8143e65a4b66807280fad97935bfdc84b75213bb75aba90b`. A runner with another hash refuses the item.

## 11. Amendments
(none yet)
