"""Reproduce default scripts/sweep.py seeds; write evaluation-only regional diagnostics.

Run from the repo: python scripts/fairness_sweep.py run 20261007
All generation stays under data/_fairness. Committed benchmark artifacts are read-only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tepat import bench, scorer
from tepat.fairness_eval import BRANCH_ROLES, SEEDS
from tepat.generator import generate_branch, load_config
import sweep

OUT = ROOT / "results" / "fairness"
DATA = ROOT / "data" / "_fairness"
PROTOCOL_VERSION = 1


def run_seed(seed: int, *, out_dir: Path = OUT, data_root: Path = DATA,
             cfg: dict | None = None, keep_data: bool = False) -> Path:
    """One resumable, atomically completed seed. A is the only training branch."""
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    sweep._no_throttle()  # Same bounded worker setup as the original sweep.
    config = copy.deepcopy(cfg if cfg is not None else load_config(ROOT / "config" / "default.yaml"))
    config["seed"] = seed
    sources = ["config/default.yaml", "src/tepat/generator.py", "src/tepat/scorer.py",
               "src/tepat/rules.py", "src/tepat/bench.py", "scripts/sweep.py"]
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    metadata = dict(protocol_version=PROTOCOL_VERSION, seed=seed, config=config, source_sha256=hashes,
                    training_random_state=seed % 2**31, branch_roles=BRANCH_ROLES,
                    final_ranking="tepat_v2_1", fairness_population="registered_observed_employers",
                    fairness_policy="final_without_random_arm", small_n_threshold=30,
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__)
    metadata = json.loads(json.dumps(metadata))  # YAML year keys become strings in JSON.
    path = out_dir / "runs" / f"seed_{seed}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["metadata"] != metadata:
            raise ValueError(f"incompatible completed run: {path}; choose a separate output directory")
        if not saved.get("groups") or len(saved.get("headlines", [])) != 21:
            raise ValueError(f"incomplete completed run: {path}")
        print(f"seed {seed}: resume completed file {path}", flush=True)
        return path
    data = data_root / f"seed_{seed}"
    start = time.monotonic()
    features, truth = {}, {}
    for branch in "ABC":
        generate_branch(copy.deepcopy(config), branch, data / branch)
        features[branch] = scorer.build_features(scorer.load_observed(data / branch), config["inspection"]["lookback_months"])
        truth[branch] = sweep._truth(data / branch)
        print(f"seed {seed}: {branch} generated and features built ({time.monotonic() - start:.1f}s)", flush=True)
    ta = truth["A"]
    train_ids = ta.sample(frac=config["inspection"]["training_sample_frac"], random_state=seed % 2**31).index
    ya, reca = ta.loc[train_ids, "is_violator"], ta.loc[train_ids, "recoverable_rp_true"]
    m1 = scorer.TepatModel(scorer.V1_FEATURES).fit(features["A"], ya, reca, "est_recoverable_v1")
    m2 = scorer.TepatModel(scorer.V2_FEATURES).fit(features["A"], ya, reca, "est_recoverable_v2")
    universe = {"sector": list(config["sectors"]), "band_obs": [b["name"] for b in config["size_bands"]],
                "kabkota": [r["name"] for r in config["kabkota"]]}
    group_rows, headlines = [], []
    for branch in "ABC":
        f = features[branch]
        rankings = {"biggest_first": scorer.biggest_first(f), "rules_only": scorer.rules_only(f),
                    "tepat_v1": scorer.rank(f, m1, "est_recoverable_v1", "tepat_v1"),
                    "tepat_v2": scorer.rank(f, m2, "est_recoverable_v2", "tepat_v2"),
                    "tepat_v2_1": scorer.rank(f, m2, "est_recoverable_v2", "tepat_v2_1", cicilan_queue=True)}
        result = bench.run(f, truth[branch], rankings, config, seed=seed + ord(branch))
        picked = bench.select_by_budget(list(rankings["tepat_v2_1"].index), f.hours, result["budget_hours"])
        groups = bench.fairness(picked, f, truth[branch], group_values=universe)
        groups = groups.assign(seed=seed, branch=branch, branch_role=BRANCH_ROLES[branch], ranking="tepat_v2_1")
        group_rows.extend(groups.astype(object).where(groups.notna(), None).to_dict("records"))
        meta = dict(run=f"seed_{seed}", seed=seed, prevalence=config["violations"]["prevalence"],
                    lookback=config["inspection"]["lookback_months"], bench_variant="base",
                    budget_frac=config["inspection"]["budget_frac"])
        headlines.extend(sweep._rows(result, meta, branch))
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(dict(metadata=metadata, groups=group_rows, headlines=headlines),
                                  ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n", encoding="utf-8")
    pending.replace(path)
    if not keep_data:
        shutil.rmtree(data)
    print(f"seed {seed}: completed in {time.monotonic() - start:.1f}s -> {path}", flush=True)
    return path


def verify_headlines(rows: pd.DataFrame, *, reference_dir: Path = ROOT / "results") -> pd.DataFrame:
    """Read-only comparisons; a missing seed or ranking fails, not disappears."""
    metrics = ["precision", "recall", "rp_per_hour_juta", "honest_flag_rate"]
    checks = []

    def add(reference, branch, ranking, metric, actual, expected, seed=None):
        passed = bool(pd.notna(actual) and np.isclose(actual, expected, rtol=1e-10, atol=1e-12))
        checks.append(dict(reference=reference, branch=branch, ranking=ranking, seed=seed,
                           metric=metric, actual=actual, expected=expected, passed=passed,
                           abs_error=abs(actual - expected) if pd.notna(actual) else None))

    if rows.duplicated(["seed", "branch", "ranking"]).any():
        raise ValueError("duplicate headline keys")
    summary = pd.read_csv(reference_dir / "sweep" / "seeds_summary.csv")
    for record in summary.to_dict("records"):
        part = rows[(rows.branch == record["branch"]) & (rows.ranking == record["ranking"])]
        add("seeds_summary.csv", record["branch"], record["ranking"], "seed_coverage", len(set(part.seed) & set(SEEDS)), len(SEEDS))
        for metric in metrics:
            for stat in ["mean", "min", "max"]:
                add("seeds_summary.csv", record["branch"], record["ranking"], f"{metric}_{stat}",
                    getattr(part[metric], stat)(), record[f"{metric}_{stat}"])
    for seed in SEEDS:
        original = pd.read_csv(reference_dir / "sweep" / "runs" / f"seed_{seed}.csv")
        for record in original[original.bench_variant == "base"].to_dict("records"):
            part = rows[(rows.seed == seed) & (rows.branch == record["branch"]) & (rows.ranking == record["ranking"])]
            for metric in metrics + ["inspected", "hits", "budget_hours", "rp_share"]:
                add("seed_runs", record["branch"], record["ranking"], metric,
                    part.iloc[0][metric] if len(part) == 1 else np.nan, record[metric], seed)
    default = json.loads((reference_dir / "results.json").read_text(encoding="utf-8"))
    for branch, result in default["branches"].items():
        for ranking, record in result["rankings"].items():
            part = rows[(rows.seed == SEEDS[0]) & (rows.branch == branch) & (rows.ranking == ranking)]
            for metric in metrics + ["inspected", "hits", "rp_share"]:
                expected = record["rp_per_hour"] / 1e6 if metric == "rp_per_hour_juta" else record[metric]
                add("results.json", branch, ranking, metric,
                    part.iloc[0][metric] if len(part) == 1 else np.nan, expected, SEEDS[0])
    return pd.DataFrame(checks)


def aggregate(*, out_dir: Path = OUT, reference_dir: Path = ROOT / "results") -> dict:
    """Publish raw rows, coverage-aware summaries and read-only reproduction checks."""
    from tepat.fairness_eval import aggregate_groups

    payloads = [json.loads(p.read_text(encoding="utf-8")) for seed in SEEDS
                if (p := out_dir / "runs" / f"seed_{seed}.json").is_file()]
    if not payloads:
        raise ValueError("no completed fairness seed runs")
    rows = pd.DataFrame([row for p in payloads for row in p["groups"]])
    headlines = pd.DataFrame([row for p in payloads for row in p["headlines"]])
    summary = aggregate_groups(rows)
    checks = verify_headlines(headlines, reference_dir=reference_dir)
    rows.to_csv(out_dir / "group_rows.csv", index=False)
    summary.to_csv(out_dir / "group_summary.csv", index=False)
    headlines.to_csv(out_dir / "headlines.csv", index=False)
    checks.to_csv(out_dir / "reproduction.csv", index=False)
    completed = [seed for seed in SEEDS if seed in set(headlines.seed)]
    config = payloads[0]["metadata"]["config"]
    universe = {"sector": list(config["sectors"]), "band_obs": [b["name"] for b in config["size_bands"]],
                "kabkota": [r["name"] for r in config["kabkota"]]}
    present_keys = set(rows[["seed", "branch", "group", "value"]].itertuples(index=False, name=None))
    missing_groups = [dict(seed=seed, branch=branch, group=group, value=value)
                      for seed in SEEDS for branch in "ABC" for group, values in universe.items() for value in values
                      if (seed, branch, group, value) not in present_keys]
    group_complete = not missing_groups and bool(summary.n_seeds_present.eq(len(SEEDS)).all())
    protocol = dict(protocol_version=PROTOCOL_VERSION, expected_seeds=SEEDS, completed_seeds=completed,
                    missing_seeds=[seed for seed in SEEDS if seed not in completed],
                    complete=completed == SEEDS and group_complete and bool(checks.passed.all()),
                    group_coverage_complete=group_complete, missing_group_rows=missing_groups,
                    branch_roles=BRANCH_ROLES, ranking="tepat_v2_1", random_arm_in_group_tables=False,
                    synthetic_only=True, fictional_geography=True, small_n_threshold=30,
                    group_rows=len(rows), summary_rows=len(summary), headline_rows=len(headlines),
                    reproduction_checks=len(checks), reproduction_passed=int(checks.passed.sum()),
                    reproduction_tolerance=dict(rtol=1e-10, atol=1e-12),
                    denominators={"pick_rate": "n_selected / n_total", "violation_rate": "n_violators / n_total",
                                  "honest_flag_rate": "n_honest_selected / n_honest"},
                    aggregation="Unweighted mean/min/max of defined per-seed rates; not confidence intervals. Counts use sum/min/max, not means. Pooled rates are ratios of count sums over present seed populations, not unique employers.",
                    missing="Missing seed/group rows do not become zero. Zero denominators are null. Per-rate seed coverage and undefined counts are explicit.",
                    stability="Group rate variation across regenerated populations, not entity rank stability; identifiers are reused, not stable entities.")
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False), flush=True)
    return protocol


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan")
    run = commands.add_parser("run")
    run.add_argument("seed", type=int, choices=SEEDS)
    run.add_argument("--keep-data", action="store_true")
    all_runs = commands.add_parser("all")
    all_runs.add_argument("--jobs", type=int, choices=[1], default=1,
                          help="One bounded compute worker on the laptop")
    all_runs.add_argument("--keep-data", action="store_true")
    commands.add_parser("aggregate")
    args = parser.parse_args()
    if args.command == "plan":
        print("\n".join(map(str, SEEDS)))
    elif args.command == "run":
        run_seed(args.seed, out_dir=args.out, keep_data=args.keep_data)
    elif args.command == "all":
        for seed in SEEDS:
            run_seed(seed, out_dir=args.out, keep_data=args.keep_data)
        if not aggregate(out_dir=args.out)["complete"]:
            raise SystemExit("Reproduction or seed coverage failed; inspect reproduction.csv")
    elif not aggregate(out_dir=args.out)["complete"]:
        raise SystemExit("Reproduction or seed coverage failed; inspect reproduction.csv")


if __name__ == "__main__":
    main()
