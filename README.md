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
uv run tepat all                               # generate A/B/C, score, benchmark  (~2 min)
uv run streamlit run app/streamlit_app.py      # demo app
uv run pytest -q                               # wall + legal-rule tests
```

## Layout

| Path | What |
|---|---|
| `config/default.yaml` | every assumption, marked as a dial (prevalence, look-alike rates, noise, inspection hours) |
| `src/tepat/rules.py` | fixed legal rules, each traceable to the Evidence Bank (E04, E05/E05c, E39, E40, E44, E55) |
| `src/tepat/generator.py` | synthetic branches with a strict wall: `truth/` (payroll, planted violations, true Rp) vs `observed/` (what BPJS would see) |
| `src/tepat/scorer.py` | reads `observed/` only: fuzzy TK↔KS matching, ~25 signals, Rp estimate, P(violation), priority, reasons |
| `src/tepat/bench.py` | compares rankings under a fixed inspection-hour budget; the only scorer-side module that reads `truth/` |
| `app/streamlit_app.py` | inspector list, case page, benchmark, fairness + random arm, worker self-check (*Cek Kepesertaan Saya*) |
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
