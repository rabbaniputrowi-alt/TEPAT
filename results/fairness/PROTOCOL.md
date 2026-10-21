# Synthetic fairness evaluation protocol

## Scope

These are diagnostics for TEPAT v2.1 under the existing inspection-hours policy, not selection quotas. All employers, planted violations, geographic names and UMK values are synthetic. The three kab/kota are fictional. These results neither establish national rates nor prove real-world fairness or the elimination of bias.

Group tables evaluate the final ranking **without** the random exploration arm. The existing app's random-arm metrics remain a separately labelled default-seed illustration. Its stratification is by observed size band only, not by sector or region. No generator, model, ranking, cicilan routing, sampling algorithm or inspection budget was changed for this evaluation.

## Reproduction

From the repository root, with project dependencies installed and `src` on `PYTHONPATH`:

```sh
python scripts/fairness_sweep.py plan
python scripts/fairness_sweep.py run 20261007
python scripts/fairness_sweep.py all --jobs 1
python scripts/fairness_sweep.py aggregate
```

Use `--out data/_fairness-check/results` **before** the subcommand to reproduce into a separate directory rather than touch the committed evidence. `run` and `all` accept `--keep-data`. One compute worker is supported deliberately. The worker uses the existing sweep's Windows power setup. Generation is isolated under `data/_fairness/seed_<seed>/`; completed seed data is removed by default. None of `data/A`, `data/B`, `data/C`, the demo slice or pre-existing benchmark outputs is regenerated or overwritten.

Seeds: `20261007`, `20261101`, `20261102`, `20261103`, `20261104`. Every seed regenerates 1,500 employers independently for each of A/B/C using `config/default.yaml`, changing only its seed.

- A is training/tuning only. Its numbers are not held-out evidence.
- B is held out, with the original held-back variants.
- C is shifted, with the original changed sector mix, look-alike scale and available variants.
- Train IDs are sampled from A using `training_sample_frac` and `random_state=seed % 2**31`.
- Both `m1` and `m2` are fitted on the original feature lists, estimates and sampled A labels.
- All five ranked methods are built exactly as in `scripts/sweep.py`; v2.1 uses `cicilan_queue=True`.
- `bench.run` uses `seed + ord(branch)`, its 50 random repetitions, the configured 10% hours budget and the unchanged 8% random-arm calculation.
- Fairness uses exactly the final ranking's budget-selected IDs, with observed sector, observed size band and observed registry-derived kab/kota. Truth supplies only planted violation labels for evaluation.

Per-seed JSON files are completed by atomic rename. A compatible completed file is reused without generation. A configuration, dependency-version or source-byte hash mismatch is rejected explicitly; use a separate output directory for a changed environment or line endings. Source hashes, full configuration, dependency versions, branch roles and the training random state are recorded in each run.

## Columns and missing values

Stable machine group keys are `sector`, `band_obs`, `kabkota`; `value` retains the original observed label. `__missing__` is an explicit missing-label bucket. Feature/truth keys and selected IDs must align one-to-one; duplicate, null, unmatched or unknown keys are rejected. Optional observed registry fallback also requires the same unique employer universe. All truth labels must be complete booleans.

| Column | Definition |
| --- | --- |
| `n_total` | All observed employers in that group, selected or not |
| `n_selected` | Employers selected within the unchanged hours budget |
| `n_violators` | Employers with planted violations |
| `n_honest` | Employers without planted violations |
| `n_honest_selected` | Selected employers without planted violations |
| `pick_rate` | `n_selected / n_total` |
| `violation_rate` | `n_violators / n_total`, the planted rate, not detected prevalence |
| `honest_flag_rate` | `n_honest_selected / n_honest`, not the honest fraction of the selected queue |

The legacy `n` column aliases `n_total`. Existing rate names are retained. Zero denominators produce undefined values (JSON null, blank CSV/UI cell), never a measured zero. Configured or categorical groups with no employers retain a row with zero counts and undefined rates. A present group with employers and no selections has a genuinely measured zero selection rate.

`small_n` flags `n_total < 30` or `n_honest < 30`. This is a descriptive caution, not a significance test. Machine warnings distinguish `small_n_total`, `small_n_honest`, `empty_group`, `no_honest`, and `missing_label`.

## Across-seed aggregation

A/B/C remain separate. Each group rate reports the unweighted mean, minimum and maximum over **defined** seed rates, plus its own number of defined seeds and undefined-present seeds. These min-to-max ranges are **not confidence intervals**. Missing seed/group rows are not imputed as zero. `n_seeds_present`, `n_seeds_expected`, `n_nonempty_seeds`, `seeds_present` and `seeds_missing` distinguish coverage from measured emptiness. Protocol completion checks the configured group universe as well as headline seed coverage.

Counts report sums and per-seed minima/maxima, never averaged denominators. Count sums refer to observations across regenerated seed populations, not unique employers. Separately named `_pooled` rates are ratios of count sums and are not presented as the unweighted seed mean. Employer IDs are reused across generated worlds; there is no claim of entity rank stability.

## Real execution and checks

The committed evaluation contains 5 completed seed files, 240 raw group rows, 48 group summaries and 105 headline rows. Every group summary has 5/5 seed coverage. `protocol.json` records complete headline and group coverage.

All **1,260** read-only reproduction checks passed at `rtol=1e-10`, `atol=1e-12`:

| Reference | Checks passed | Maximum absolute error |
| --- | ---: | ---: |
| Default seed against `results/results.json` | 147/147 | 0 |
| All seeds against `results/sweep/runs/seed_*.csv`, base variant only | 840/840 | 1.1368683772161605e-13 |
| Mean/min/max and coverage against `results/sweep/seeds_summary.csv` | 273/273 | 1.7763568394002505e-15 |

Checks cover every existing ranking, not only the final model. Aggregation compares precision, recall, Rp/hour and honest flag rate, with per-seed selected counts, hits, budget hours and recoverable share also checked. The original benchmark artifacts are read-only.

## Files

- `runs/seed_<seed>.json`: actual per-seed raw groups, all headline rankings and provenance metadata.
- `group_rows.csv`: concatenated raw group/count/rate rows.
- `group_summary.csv`: coverage-aware rate summaries, count sums/ranges and separately named pooled rates.
- `headlines.csv`: all existing ranking metrics for all five seeds and three branches.
- `reproduction.csv`: every reference comparison, tolerance result and absolute error.
- `protocol.json`: machine-readable scope, denominators, coverage and reproduction status.

The app reads the small per-seed files and recomputes the summary with the same pure evaluation helper. It shows committed default-config populations in both demo and local modes, not live follow-up outcomes or an evaluation of arbitrary replacement local data. It does not read truth into scoring or retrain a model.
