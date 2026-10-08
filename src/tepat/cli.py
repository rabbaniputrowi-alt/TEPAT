"""TEPAT command line: generate -> features -> train on A -> rank -> benchmark.

    uv run tepat all                  # full pipeline, writes data/ and results/
    uv run tepat all --employers 300  # quick run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import bench, scorer
from .generator import generate_branch, load_config

ROOT = Path(__file__).resolve().parents[2]


def _truth(branch_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(branch_dir / "truth" / "employers_truth.parquet").set_index("employer_id")


def cmd_all(args) -> None:
    cfg = load_config(args.config)
    if args.employers:
        cfg["n_employers"] = args.employers
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    branches = ["A", "B", "C"]
    log = {}
    for b in (args.branches or branches):
        t = time.time()
        if not args.skip_generate:
            log[b] = generate_branch(cfg, b, data / b)
            print(f"[generate] {b}: {log[b]} ({time.time() - t:.0f}s)", flush=True)

    if args.generate_only:
        return
    feats = {}
    for b in branches:
        t = time.time()
        fp = out / f"features_{b}.parquet"
        if args.reuse_features and fp.exists():
            feats[b] = pd.read_parquet(fp)
        else:
            feats[b] = scorer.build_features(scorer.load_observed(data / b), cfg["inspection"]["lookback_months"])
            feats[b].to_parquet(fp)
        print(f"[features] {b}: {feats[b].shape} ({time.time() - t:.0f}s)", flush=True)
        if args.features_only and b == branches[-1]:
            return

    # Train on branch A: a random sample stands in for past inspection outcomes.
    TA = _truth(data / "A")
    rng = np.random.default_rng(cfg["seed"])
    train_ids = TA.sample(frac=cfg["inspection"]["training_sample_frac"], random_state=int(cfg["seed"]) % 2**31).index
    yA, recA = TA.loc[train_ids, "is_violator"], TA.loc[train_ids, "recoverable_rp_true"]
    m1 = scorer.TepatModel(scorer.V1_FEATURES).fit(feats["A"], yA, recA, "est_recoverable_v1")
    m2 = scorer.TepatModel(scorer.V2_FEATURES).fit(feats["A"], yA, recA, "est_recoverable_v2")
    coefs = dict(zip(scorer.V2_FEATURES, m2.clf[-1].coef_[0].round(3).tolist()))

    results = {"config_seed": cfg["seed"], "n_employers": cfg["n_employers"], "train_size": len(train_ids),
               "v2_coefficients": coefs, "branches": {}}
    tables = []
    for b in branches:
        F, T = feats[b], _truth(data / b)
        r_v1 = scorer.rank(F, m1, "est_recoverable_v1", "tepat_v1")
        r_v2 = scorer.rank(F, m2, "est_recoverable_v2", "tepat_v2")
        r_v21 = scorer.rank(F, m2, "est_recoverable_v2", "tepat_v2_1", cicilan_queue=True)
        rankings = {"biggest_first": scorer.biggest_first(F), "rules_only": scorer.rules_only(F),
                    "tepat_v1": r_v1, "tepat_v2": r_v2, "tepat_v2_1": r_v21}
        res = bench.run(F, T, rankings, cfg, seed=int(cfg["seed"]) + ord(b))
        results["branches"][b] = res
        scorer.explain(F, r_v21).to_parquet(out / f"ranked_{b}.parquet")
        tab = bench.summary_table(res)
        tab.insert(0, "branch", b)
        tables.append(tab)
        print(f"\n=== Branch {b} ({'tune' if b == 'A' else 'test' if b == 'B' else 'shifted test'}) "
              f"| budget {res['budget_hours']:.0f} h | violators {res['violators']} ({res['prevalence']:.0%})")
        print(tab.drop(columns="branch").to_string(index=False))
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    pd.concat(tables).to_csv(out / "summary.csv", index=False)
    print(f"\nwrote {out / 'results.json'} and {out / 'summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="tepat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all", help="generate, score, benchmark")
    a.add_argument("--config", default=str(ROOT / "config" / "default.yaml"))
    a.add_argument("--data", default=str(ROOT / "data"))
    a.add_argument("--out", default=str(ROOT / "results"))
    a.add_argument("--employers", type=int, default=None)
    a.add_argument("--skip-generate", action="store_true")
    a.add_argument("--branches", nargs="*", default=None, help="generate only these branches")
    a.add_argument("--generate-only", action="store_true")
    a.add_argument("--features-only", action="store_true")
    a.add_argument("--reuse-features", action="store_true")
    args = ap.parse_args()
    {"all": cmd_all}[args.cmd](args)


if __name__ == "__main__":
    main()
