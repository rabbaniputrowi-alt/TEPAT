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

## Case follow-up and worker-report triage

The inspector workspace includes **Tindak lanjut** and **Laporan pekerja**. Worker reports enter a
branch-scoped inbox for review, investigation notes, dated history and an explicit resolution. Repeated
reports from the same reporter about the same employer and issue in the same branch retain one ticket
and record the duplicate submission. Reports are allegations, not inspection findings; they do not change
rankings or create cases automatically.

An inspector can explicitly create or link a follow-up case. Existing cases, including closed cases,
are reused without resetting their outcome. Resolving a report does not close its company case.
Case outcomes can be exported as labels for a later training run; the app never retrains itself.

Storage follows the existing case-store mode:
- Full local data defaults to a gitignored `results/cases.sqlite` database for cases, reports and their
  histories. It survives app and Python-process restarts while that file remains available.
- The committed demo slice defaults to a private, seeded in-memory database per browser session.
  Visitors cannot see each other's edits. New sessions and app restarts reset that sandbox.
- `TEPAT_CASES=persistent` enables local file storage when using the slice;
  `TEPAT_CASES=sandbox` forces private session storage. `TEPAT_RESULTS` relocates the SQLite file.
  File storage on an ephemeral cloud container is not durable hosting; keep the public demo in sandbox mode.

Only a full SHA-256 reporter hash is stored in the report repository, not a raw NIK. Unsalted hashing is
not production anonymization. Use synthetic records only: authentication, production access controls and
encryption are not implemented. Nothing is delivered to BPJS or Mobile JKN.

Old `results/tips.csv` files are preserved but not imported or counted. Their schema lacks a branch,
so assigning those reports safely requires an explicit migration decision.

## Draf dokumen (case view)

The case page drafts three documents in Bahasa Indonesia from observed evidence only (scorer output,
features, registry; never the answer key): an internal inspection brief, a compliance letter offering
self-correction within a deadline (14 calendar days by default), and an instalment offer for cases in the
cicilan queue. Every draft is labelled as a synthetic draft that is not sent. The inspector reviews it,
copies the plain text or downloads a DOCX, and approves it. With a follow-up case, approval is recorded
in its history and can suggest a next status; it never sends a letter or changes status automatically.

A deterministic template always works. An optional rewrite through any OpenAI-compatible
`/chat/completions` endpoint is enabled only when all three of `TEPAT_LLM_API_KEY`, `TEPAT_LLM_BASE_URL`
(for example `https://host/v1`) and `TEPAT_LLM_MODEL` are set as environment variables or in
`.streamlit/secrets.toml`. No key ships with the repo. A fact checker rejects any rewrite that adds a number,
date, name or legal citation not present in the template or the evidence, and the template is used instead.

## Layout

| Path | What |
|---|---|
| `config/default.yaml` | every assumption, marked as a dial (prevalence, look-alike rates, noise, inspection hours) |
| `src/tepat/rules.py` | fixed legal rules, each traceable to the Evidence Bank (E04, E05/E05c, E39, E40, E44, E55) |
| `src/tepat/generator.py` | synthetic branches with a strict wall: `truth/` (payroll, planted violations, true Rp) vs `observed/` (what BPJS would see) |
| `src/tepat/scorer.py` | reads `observed/` only: fuzzy TK↔KS matching, ~25 signals, Rp estimate, P(violation), priority, reasons |
| `src/tepat/bench.py` | compares rankings under a fixed inspection-hour budget; the only scorer-side module that reads `truth/` |
| `app/streamlit_app.py` | priority queue, employer case, follow-up board, report inbox, registration list, benchmark, fairness, worker self-check |
| `src/tepat/cases.py`, `src/tepat/reports.py` | shared SQLite case/report workflow and separate dated histories |
| `app/reports_panel.py` | branch-scoped report inbox, review, case linkage and resolution |
| `src/tepat/fairness_eval.py`, `app/fairness_panel.py` | denominator-aware group diagnostics and cross-seed display |
| `scripts/fairness_sweep.py`, `results/fairness/` | resumable five-seed evaluation, raw counts, summaries and reproduction checks |
| `src/tepat/drafting.py` | evidence-only drafts (brief, compliance letter, cicilan offer), fact checker, optional LLM rewrite, DOCX |
| `app/drafting_panel.py` | the case page's *Draf dokumen* panel: pick, review, copy, download, approve |
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

## Expanded fairness evaluation

**Keadilan & jalur acak** shows sector, observed size-band and fictional kab/kota diagnostics.
Each group includes total employers, selected employers, planted violators, honest employers and selected
honest employers, alongside the three rates and their denominators. Zero-denominator rates are undefined,
not measured zero. Small groups are labelled for caution.

The committed evaluation regenerates the five default-config seeds, trains only on A and keeps A (tuning),
B (held out) and C (shifted) separate. It reports per-seed rates, unweighted mean/min/max, count sums/ranges
and seed coverage. These ranges are not confidence intervals or company-rank stability. Group tables
exclude the random arm; the separately labelled default-seed arm stratifies by observed size only.

The panel reads saved default-config evaluation in both local and demo modes, not arbitrary replacement
local data or inspector outcomes. All geographic names are fictional. This demonstrates a diagnostic
workflow, not fairness on real populations or the elimination of bias.

Reproduction commands and the complete protocol are in `results/fairness/PROTOCOL.md`. To rerun without
overwriting committed evidence:

```bash
uv run python scripts/fairness_sweep.py --out data/_fairness-check/results all --jobs 1
```

The evaluation records 1,260 successful checks against the existing benchmark and sweep. Original rankings,
generator/scorer behavior and previously committed benchmark artifacts are unchanged.

**Known weaknesses:** workers hidden from both agencies (1b) and peer-relative wage under-reporting (2a/2b)
have low recall. The main list covers registered employers only; the separate registration pass has its own
synthetic benchmark and a modest gain over rules. Five-seed and sensitivity results live in `results/sweep/`;
all results remain synthetic method evidence, not measured operational recovery or real-world validation.
