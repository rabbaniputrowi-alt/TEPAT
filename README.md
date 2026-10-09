# TEPAT: Pemeriksaan Kepatuhan Tepat Sasaran

Prototype for **Healthkathon 2026 (BPJS Kesehatan), category Pemberi Kerja**. Team Johannes Kepler.

TEPAT ranks employers for BPJS Kesehatan compliance inspection by the **expected rupiah recovered per inspection hour**,
using only data that BPJS Kesehatan already holds or receives from BPJS Ketenagakerjaan. Every flag comes with
plain-Indonesian reasons, the legal basis, a document request list and a suggested action. The inspector decides.

> **Everything here is synthetic.** No real company, worker, NIK or BPJS data is used. NIKs start with `99`
> (not a real province code). UMK values belong to fictional kab/kota. Results are *method* numbers on synthetic
> data whose assumptions are dials in `config/default.yaml`.

## Run

```bash
uv sync
uv run streamlit run app/streamlit_app.py      # demo app; a fresh clone runs on the committed demo/ slice
uv run tepat all                               # optional: regenerate A/B/C, score, benchmark (~2 min)
uv run pytest -q                               # wall, legal-rule, app and demo-slice tests
```

The app reads the full local run (`data/<branch>/` + `results/*.parquet`, both gitignored) when it exists and
falls back to `demo/` otherwise. `TEPAT_DEMO=1` forces the slice.

## Demo slice (`demo/`, about 7 MB)

What a fresh clone and Streamlit Community Cloud run on. Built from the full run by
`uv run python scripts/make_demo_slice.py`; `demo/manifest.json` records the selection and a SHA-256 per file.

- **All 1,500 employers per branch** for everything employer-level: registry, billing, ranked output, features,
  `employers_truth`. The queue, case, benchmark and fairness views show exactly what the full run shows.
- **Worker-level tables for 200 employers per branch**: the top 40 of the inspection queue, the whole
  installment queue, the best-ranked employer for each worker scenario, then a size-stratified random fill.
  The cut is closed under NIK (a kept worker keeps every row at every employer), so *Cek Kepesertaan Saya*
  answers exactly as on the full data for every NIK it holds. Workers of other employers are absent, and the
  view says so.
- Left out: `truth/payroll_truth` and `truth/labels` (the app never reads them).
- `tests/test_demo.py` checks the hashes, renders every view on every branch from the slice, and (when the full
  run is present) checks the slice frame by frame against it.

## Deploy on Streamlit Community Cloud

1. share.streamlit.io, **Create app**, then pick repo `rabbaniputrowi-alt/TEPAT`, branch `main`,
   main file `app/streamlit_app.py`.
2. **Advanced settings**: Python **3.11** (matches `.python-version` and the lockfile). Python can only be
   changed later by deleting and redeploying.
3. Dependencies come from `app/requirements.txt`, exported from `uv.lock`. Community Cloud looks in the
   entrypoint folder before the repo root, so this file wins over the root `uv.lock`. After changing
   dependencies, re-export with
   `uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt > app/requirements.txt`
   (a test fails if the two drift).
4. A private repo gives a private app (one per account). Invite viewers by email, or switch the app to public
   under **Share** when judges need the link.

Worker reports from the demo button are written to `results/tips.csv` on the app's container: shared by all
visitors and lost when the app restarts. They are demo records only.

## Layout

| Path | What |
|---|---|
| `config/default.yaml` | every assumption, marked as a dial (prevalence, look-alike rates, noise, inspection hours) |
| `src/tepat/rules.py` | fixed legal rules, each traceable to the Evidence Bank (E04, E05/E05c, E39, E40, E44, E55) |
| `src/tepat/generator.py` | synthetic branches with a strict wall: `truth/` (payroll, planted violations, true Rp) vs `observed/` (what BPJS would see) |
| `src/tepat/scorer.py` | reads `observed/` only: fuzzy TK↔KS matching, ~25 signals, Rp estimate, P(violation), priority, reasons |
| `src/tepat/bench.py` | compares rankings under a fixed inspection-hour budget; the only scorer-side module that reads `truth/` |
| `app/streamlit_app.py` | inspector list, case page, benchmark, fairness + random arm, worker self-check (*Cek Kepesertaan Saya*) |
| `app/requirements.txt` | pinned runtime dependencies for Streamlit Community Cloud (exported from `uv.lock`) |
| `demo/` | committed demo slice; `scripts/make_demo_slice.py` rebuilds it |
| `tests/test_wall.py` | the scorer never imports the generator or reads `truth/`; legal rules hold |

## Benchmark protocol

- **A** tune (40% sample = "past inspection outcomes" for training), **B** test (new seed), **C** shifted test
  (different sector mix, 1.5× look-alike rates, variants 1c/1d/2b never seen in A).
- Budget = hours to inspect ~10% of the branch's total inspection hours.
- Rankings: random, biggest-first, rules-only (TK>KS headcount, wage<UMK for medium/large only, overdue; tie-break by size),
  TEPAT v1 (core signals), v2 (+ cross-agency bundles), **v2.1** (+ can't-pay employers routed to an installment queue).
- v2.1 was added after B/C had been viewed once; it was tuned on A only, and v2 stays in the table.

## Results (seed 20261007, 1,500 employers per branch, budget ≈ 10% of hours)

| Branch | Ranking | Precision | Rp per inspection hour | Honest employers flagged |
|---|---|---|---|---|
| B (test) | random | 21% | Rp1.0 jt | 10.0% |
| B (test) | rules only | 40% | Rp3.3 jt | 4.2% |
| B (test) | **TEPAT v2.1** | **95%** | **Rp7.6 jt** | **0.4%** |
| C (shifted) | rules only | 36% | Rp7.4 jt | 4.9% |
| C (shifted) | **TEPAT v2.1** | **95%** | **Rp11.2 jt** | **0.4%** |

Full tables: `results/summary.csv`, `results/results.json` (recall per variant, look-alike flag rates, fairness, Rp-range calibration).

**Known weaknesses** (from the benchmark itself): workers hidden from *both* agencies (1b) and peer-relative wage
under-reporting (2a/2b) have low recall; employers that never registered at all are invisible to TEPAT;
one seed only, so no confidence intervals yet.
