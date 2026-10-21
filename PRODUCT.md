# TEPAT

A Healthkathon 2026 prototype for BPJS Kesehatan employer-compliance inspection. Team: Johannes Kepler.

## Product truth
- Inspectors rank registered employers by estimated recoverable rupiah per inspection hour, inspect evidence, and choose an action.
- The scorer uses observed synthetic records; answer keys are isolated in explicitly labelled demo expanders and evaluation views.
- Eight views: priority queue, employer case, case follow-up, worker-report inbox, registration list, benchmark, fairness/random arm and worker membership check/report.
- Can't-pay employers have a separate cicilan queue. The human inspector makes the decision.
- Cases record inspectors, status, follow-up dates, notes, history and outcomes. Reports have branch-scoped triage, duplicate handling, review notes and a separate resolution; explicit case linkage never resets an existing case outcome or changes model rankings.
- Local file mode persists cases and reports in a gitignored SQLite database. The public demo gives each visitor a private seeded in-memory sandbox that resets with the session or app restart.
- Reports store a full SHA-256 reporter hash, not a raw NIK. This is not production anonymization. Old tips.csv files are preserved but not imported because their schema lacks a branch. Nothing is sent to BPJS or Mobile JKN.
- Evidence-grounded documents are reviewed and exported, not sent. Approval records a case event when a case exists and only suggests the next status.
- Fairness diagnostics include sector, observed size band and fictional kab/kota, explicit counts/denominators, and five-seed mean/min/max with coverage. These are not confidence intervals, company-rank stability or proof of real-world fairness. Group tables exclude the separately labelled random arm.
- All firms, workers, NIKs, geographic names, UMK values and benchmarks are synthetic. Saved evaluation uses the default configuration, not arbitrary replacement local data or measured operational impact. Production authentication, access controls, encryption and real system connections are not implemented.

## Implementation constraints
Python 3.11, Streamlit 1.65, Altair, pandas; code lives outside the Obsidian vault. Preserve the generator, scorer, benchmark results, legal caveats, and the truth/observed wall during UI changes.

## Usage scene
A desktop-first inspector workspace used in ordinary office light, plus a mobile-friendly worker self-check. A clear task path matters more than decorative motion. Indonesian interface copy; no emojis.
