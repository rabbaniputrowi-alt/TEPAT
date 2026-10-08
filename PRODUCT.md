# TEPAT

A Healthkathon 2026 prototype for BPJS Kesehatan employer-compliance inspection. Team: Johannes Kepler.

## Product truth
- Inspectors rank registered employers by estimated recoverable rupiah per inspection hour, inspect evidence, and choose an action.
- The scorer uses observed synthetic records; answer keys are isolated in explicitly labelled demo expanders and evaluation views.
- Five views: priority queue, employer case, benchmark, fairness/random arm, worker membership check/report.
- Can't-pay employers have a separate cicilan queue. The human inspector makes the decision.
- Worker reporting writes a local demo ticket, never a real BPJS report. Only a truncated NIK hash is stored.
- All firms, workers, NIKs, UMK values, and benchmarks are synthetic. Results currently represent one seed, not demonstrated operational impact.

## Implementation constraints
Python 3.11, Streamlit 1.65, Altair, pandas; code lives outside the Obsidian vault. Preserve the generator, scorer, benchmark results, legal caveats, and the truth/observed wall during UI changes.

## Usage scene
A desktop-first inspector workspace used in ordinary office light, plus a mobile-friendly worker self-check. A clear task path matters more than decorative motion. Indonesian interface copy; no emojis.
