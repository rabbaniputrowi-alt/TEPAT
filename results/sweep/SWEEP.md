# TEPAT multi-seed and sensitivity sweep (2026-10-10)

Synthetic method numbers. Every run regenerates branches A, B and C from scratch, trains on a 40% sample of A
exactly as `uv run tepat all` does, and benchmarks B (held-out test) and C (shifted test). Rebuild with
`uv run python scripts/sweep.py all --jobs 3` then `uv run python scripts/sweep.py aggregate`.

- **20 regenerated runs**: 5 seeds at the default config, plus prevalence 5%, 10% and 35% and lookback 6 and
  24 months, each on 3 seeds. Budget (5%, 20%) and inspection-hour schemes (flat, steep) need no regeneration;
  they are computed on all 5 seed runs, and the sensitivity table averages the same 3 seeds as the other dials.
- **Reproduction check**: the default seed (20261007) reproduces the committed `results/summary.csv` to its
  3-decimal rounding.
- Ratios are reported as ratio of means across seeds, with the per-seed range. A mean of per-seed ratios is
  misleading when a baseline recovers almost nothing in one seed (prevalence 5%, branch C: 216x in one seed).

## Five seeds, default config (budget 10% of inspection hours)

| Branch | Ranking | Precision | Recall | Rp per inspection hour | Honest employers flagged |
|---|---|---|---|---|---|
| B (test) | Random | 20% (19 to 21) | 10% | Rp1,1 jt (0,9 to 1,3) | 10,0% |
| B | Rules only | 35% (28 to 41) | 9% | Rp3,9 jt (1,9 to 4,9) | 4,0% (3,5 to 4,5) |
| B | TEPAT v1 | 93% (89 to 97) | 31% | Rp8,3 jt (6,8 to 10,4) | 0,5% |
| B | TEPAT v2 | 88% (84 to 93) | 30% | Rp8,5 jt (7,0 to 10,1) | 1,0% |
| B | **TEPAT v2.1** | **95% (93 to 97)** | **34% (32 to 36)** | **Rp8,9 jt (7,1 to 11,1)** | **0,4% (0,2 to 0,7)** |
| C (shifted) | Rules only | 32% (28 to 39) | 9% | Rp4,0 jt (1,8 to 7,4) | 4,8% |
| C | **TEPAT v2.1** | **94% (90 to 96)** | **35% (28 to 43)** | **Rp8,5 jt (6,0 to 11,2)** | **0,5% (0,4 to 0,9)** |

Ranges are min to max across the 5 seeds.

| Branch | TEPAT v2.1 vs rules only | vs random | v2 vs v1 | v2.1 vs v1 |
|---|---|---|---|---|
| B | **2,3x** (per seed 1,9 to 3,7) | **8,0x** (7,6 to 8,4) | +3% (-3 to +7) | +7% (+5 to +9) |
| C | **2,1x** (1,5 to 3,3) | **7,7x** (6,7 to 8,3) | +6% (-2 to +14) | +7% (-1 to +16) |

The committed single-seed figure (B: 2,3x rules, Rp7,6 jt) sits at the low end of the seed range for Rp per
hour and at the mean for the ratio. **It is a fair figure to quote, with the range beside it.**

## Sensitivity (one dial at a time, mean of 3 seeds)

| Dial | Value | Branch | TEPAT v2.1 Rp/h | Rules Rp/h | Ratio (seed range) | TEPAT precision | Rules precision | Honest flagged, TEPAT vs rules |
|---|---|---|---|---|---|---|---|---|
| Default | 20%, 12 mo, 10%, 3/6/12/24 h | B | 8,0 jt | 3,4 jt | 2,4x (1,9 to 3,7) | 95% | 34% | 0,5% vs 4,2% |
| Prevalence | 5% | B | 4,0 jt | 2,5 jt | 1,6x (1,3 to 1,8) | 34% | 16% | 5,2% vs 3,9% |
| Prevalence | 10% | B | 4,7 jt | 1,7 jt | 2,7x (2,3 to 4,0) | 64% | 16% | 3,8% vs 4,2% |
| Prevalence | 35% | B | 11,9 jt | 5,9 jt | 2,0x (1,7 to 2,3) | 98% | 58% | 0,2% vs 3,5% |
| Lookback | 6 months | B | 5,2 jt | 2,3 jt | 2,3x (1,8 to 3,5) | 95% | 34% | 0,5% vs 4,2% |
| Lookback | 24 months | B | 13,6 jt | 5,5 jt | 2,5x (2,0 to 3,9) | 95% | 34% | 0,5% vs 4,2% |
| Budget | 5% of hours | B | 12,8 jt | 4,5 jt | 2,8x (2,1 to 3,9) | 98% | 38% | 0,1% vs 2,5% |
| Budget | 20% of hours | B | 4,6 jt | 2,8 jt | 1,6x (1,5 to 1,7) | 79% | 30% | 4,4% vs 7,8% |
| Hours | flat 6/6/6/6 | B | 9,3 jt | 6,7 jt | 1,4x (1,3 to 1,5) | 80% | 30% | 2,5% vs 8,8% |
| Hours | steep 2/6/16/48 | B | 6,3 jt | 2,4 jt | 2,7x (2,3 to 3,6) | 94% | 36% | 0,6% vs 3,7% |

Branch C follows the same pattern (see `sensitivity.csv`). At prevalence 5% on C the ratio of means is 2,5x.

### What the sweep says

1. **TEPAT beats the rules baseline on Rp per hour in all 80 run, setting and branch combinations on B and C**
   (none below 1x on any branch). The worst single case is 1,18x (flat hours, branch C, seed 20261103).
2. **The edge is largest when inspections are scarce and costly**: a 5% budget or a steep hour scale gives
   2,7x to 2,8x. With a 20% budget the two converge (1,6x) because both reach the easy cases.
3. **Flat inspection hours cut the edge to 1,4x.** Part of TEPAT's advantage is ranking by rupiah per hour,
   so it matters how much a large employer costs to inspect compared with a small one. The proposal should
   say this assumption drives the size of the gain.
4. **Low prevalence is the weak spot.** At 5% violators, a 10% hour budget cannot be filled with violators,
   so precision falls to about a third and TEPAT flags slightly more honest employers than the rules
   (5,2% vs 3,9%). It still recovers 1,6x the rupiah per hour. In practice the budget would be cut, not
   spent on honest employers.
5. **Lookback scales rupiah, not ranking quality**: precision and honest-flag rates are unchanged from 6 to 24
   months; Rp per hour scales with the window for both methods.
6. **v2's cross-agency bundles add +3% (B) and +6% (C) over v1 on Rp per hour, and the seed range crosses
   zero.** The cicilan routing in v2.1 is the consistent gain (+7%). The v2 bundles matter for reasons and
   actions, not for ranking.

## Weak spots, 5 seeds (TEPAT v2.1 recall, branch B unless noted)

- **1b** (missing at both agencies): 8% (5 to 13). **2a** peer-relative wage under-reporting: 21% (16 to 31).
  **2b** wage scaling, never seen in training (C only): 9% (6 to 13). **5b** early deregistration: 21% (8 to
  33). **5d**: 11% (0 to 33).
- Strong: **1a** 54%, **3c** 55%, **1d** piggyback (C only, unseen) 58%.
- **C11 cash-stressed look-alikes** remain the most-flagged honest group: 13% on B (4 to 20), 7% on C.
- **Rp range calibration**: 89% of true gaps fall inside the stated 10 to 90% range (85 to 93). The target is
  about 80%, so the range is slightly too wide on every seed.
- **Random arm**: 15 or 16 picks per branch; its violation rate swings from 6% to 33% across seeds. It is
  an unbiased check, but too small to estimate prevalence from one cycle.

## Files

`runs/<run>.csv` one file per run; `all_rows.csv` everything; `seeds_summary.csv`, `seeds_ratios.csv`,
`sensitivity.csv`, `variant_recall.csv` aggregates.
