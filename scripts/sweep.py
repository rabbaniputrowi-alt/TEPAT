"""Multi-seed and sensitivity sweep for the TEPAT benchmark.

Each *run* regenerates branches A/B/C under one config (seed, prevalence, lookback), trains on A exactly
as `tepat all` does, then benchmarks several *bench variants* that do not need regeneration (inspection
budget, inspection-hour scheme). Dials are varied one at a time around the default config.

    uv run python scripts/sweep.py plan                   # list runs
    uv run python scripts/sweep.py run seed_20261007      # one run -> results/sweep/runs/<name>.csv
    uv run python scripts/sweep.py all --jobs 2           # every missing run, in parallel subprocesses
    uv run python scripts/sweep.py aggregate              # -> results/sweep/*.csv + SWEEP.md

Regenerated data goes to data/_sweep/<run>/ (gitignored) and is deleted after the run unless --keep-data.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tepat import bench, scorer  # noqa: E402
from tepat.generator import generate_branch, load_config  # noqa: E402

OUT = ROOT / "results" / "sweep"
DATA = ROOT / "data" / "_sweep"

SEEDS = [20261007, 20261101, 20261102, 20261103, 20261104]  # the first is the committed default run
SENS_SEEDS = SEEDS[:3]
PREVALENCE = [0.05, 0.10, 0.35]  # default 0.20
LOOKBACK = [6, 24]  # default 12 (sourced range 6-24, E55)
BUDGETS = [0.05, 0.20]  # default 0.10
HOURS = {  # hours per size band 2-9 / 10-49 / 50-249 / 250+; default 3/6/12/24
    "flat": [6, 6, 6, 6],
    "steep": [2, 6, 16, 48],
}
RANKINGS = ["random", "biggest_first", "rules_only", "tepat_v1", "tepat_v2", "tepat_v2_1", "tepat_v2_1_with_random_arm"]


def plan() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for s in SEEDS:
        runs[f"seed_{s}"] = dict(seed=s, prevalence=0.20, lookback=12)
    for p in PREVALENCE:
        for s in SENS_SEEDS:
            runs[f"prev{int(p * 100):02d}_{s}"] = dict(seed=s, prevalence=p, lookback=12)
    for lb in LOOKBACK:
        for s in SENS_SEEDS:
            runs[f"lb{lb:02d}_{s}"] = dict(seed=s, prevalence=0.20, lookback=lb)
    return runs


def _no_throttle() -> None:
    """Windows on battery throttles background processes (EcoQoS) to ~1/3 speed. Opt this process out
    and raise its priority. No-op elsewhere; failures are ignored (the run is only slower)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        h = k.GetCurrentProcess()
        s = State(1, 1, 0)  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED controlled, state off
        k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s))  # 4 = ProcessPowerThrottling
        k.SetPriorityClass(h, 0x80)  # HIGH_PRIORITY_CLASS
    except Exception:
        pass


def _truth(d: Path) -> pd.DataFrame:
    return pd.read_parquet(d / "truth" / "employers_truth.parquet").set_index("employer_id")


def _rows(res: dict, meta: dict, branch: str) -> list[dict]:
    rows = []
    for name, r in res["rankings"].items():
        row = dict(**meta, branch=branch, ranking=name, prevalence_obs=res["prevalence"], budget_hours=res["budget_hours"],
                   inspected=r["inspected"], hits=r["hits"], precision=r["precision"], recall=r["recall"],
                   rp_per_hour_juta=r["rp_per_hour"] / 1e6, rp_share=r["rp_share"],
                   honest_flag_rate=r["honest_flag_rate"])
        if name == "tepat_v2_1":
            row["rp_range_coverage"] = res.get("rp_range_coverage")
            for v, x in r["recall_by_variant"].items():
                row[f"recall_{v}"] = x
            for c, x in r["lookalike_flag_rate"].items():
                row[f"flag_{c}"] = x
        if name == "tepat_v2_1_with_random_arm":
            row["random_arm_size"] = r["random_arm_size"]
            row["random_arm_violation_rate"] = r["random_arm_violation_rate"]
        rows.append(row)
    return rows


def run_one(name: str, keep_data: bool = False) -> Path:
    _no_throttle()
    spec = plan()[name]
    cfg = load_config(str(ROOT / "config" / "default.yaml"))
    cfg["seed"] = spec["seed"]
    cfg["violations"]["prevalence"] = spec["prevalence"]
    cfg["inspection"]["lookback_months"] = spec["lookback"]
    data = DATA / name
    t0 = time.time()
    feats, truth = {}, {}
    for b in ["A", "B", "C"]:
        generate_branch(copy.deepcopy(cfg), b, data / b)
        feats[b] = scorer.build_features(scorer.load_observed(data / b), spec["lookback"])
        truth[b] = _truth(data / b)
        print(f"[{name}] {b} generated + features ({time.time() - t0:.0f}s)", flush=True)

    # Train on A exactly as cli.cmd_all does.
    TA = truth["A"]
    train_ids = TA.sample(frac=cfg["inspection"]["training_sample_frac"], random_state=int(cfg["seed"]) % 2**31).index
    yA, recA = TA.loc[train_ids, "is_violator"], TA.loc[train_ids, "recoverable_rp_true"]
    m1 = scorer.TepatModel(scorer.V1_FEATURES).fit(feats["A"], yA, recA, "est_recoverable_v1")
    m2 = scorer.TepatModel(scorer.V2_FEATURES).fit(feats["A"], yA, recA, "est_recoverable_v2")

    variants = [("base", 0.10, None)]
    if name.startswith("seed_"):  # budget and hours dials need no regeneration: evaluate them on the seed runs only
        variants += [(f"budget{int(x * 100):02d}", x, None) for x in BUDGETS] \
            + [(f"hours_{k}", 0.10, v) for k, v in HOURS.items()]
    rows = []
    for b in ["A", "B", "C"]:
        for vname, budget, hours in (variants if b != "A" else variants[:1]):
            F = feats[b]
            if hours is not None:
                F = F.copy()
                band_hours = dict(zip(["2-9", "10-49", "50-249", "250+"], hours))
                F["hours"] = F["band_obs"].map(band_hours).astype(float)
            c = copy.deepcopy(cfg)
            c["inspection"]["budget_frac"] = budget
            rankings = {"biggest_first": scorer.biggest_first(F), "rules_only": scorer.rules_only(F),
                        "tepat_v1": scorer.rank(F, m1, "est_recoverable_v1", "tepat_v1"),
                        "tepat_v2": scorer.rank(F, m2, "est_recoverable_v2", "tepat_v2"),
                        "tepat_v2_1": scorer.rank(F, m2, "est_recoverable_v2", "tepat_v2_1", cicilan_queue=True)}
            res = bench.run(F, truth[b], rankings, c, seed=int(cfg["seed"]) + ord(b))
            meta = dict(run=name, seed=spec["seed"], prevalence=spec["prevalence"], lookback=spec["lookback"],
                        bench_variant=vname, budget_frac=budget)
            rows += _rows(res, meta, b)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    path = OUT / "runs" / f"{name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    if not keep_data:
        shutil.rmtree(data, ignore_errors=True)
    print(f"[{name}] done in {time.time() - t0:.0f}s -> {path}", flush=True)
    return path


def run_all(jobs: int, keep_data: bool) -> None:
    todo = [n for n in plan() if not (OUT / "runs" / f"{n}.csv").exists()]
    print(f"{len(todo)} runs to do, {jobs} at a time", flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    active: list[tuple[str, subprocess.Popen]] = []
    failed = []
    while todo or active:
        while todo and len(active) < jobs:
            n = todo.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "run", n] + (["--keep-data"] if keep_data else [])
            log = open(OUT / "logs" / f"{n}.log", "w", encoding="utf-8")
            active.append((n, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)))
            print(f"started {n}", flush=True)
        time.sleep(5)
        for n, p in list(active):
            if p.poll() is not None:
                active.remove((n, p))
                print(f"finished {n} rc={p.returncode}", flush=True)
                if p.returncode:
                    failed.append(n)
    print("failed:", failed or "none", flush=True)


# --------------------------------------------------------------------------- aggregation
KEY = ["precision", "recall", "rp_per_hour_juta", "honest_flag_rate"]


def _ratio_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per run x branch x variant: TEPAT v2.1 relative to rules-only and random."""
    p = df.pivot_table(index=["run", "seed", "prevalence", "lookback", "bench_variant", "branch"],
                       columns="ranking", values="rp_per_hour_juta").reset_index()
    p["x_rules"] = p["tepat_v2_1"] / p["rules_only"]
    p["x_random"] = p["tepat_v2_1"] / p["random"]
    p["v21_minus_v1"] = p["tepat_v2_1"] / p["tepat_v1"] - 1
    p["v2_minus_v1"] = p["tepat_v2"] / p["tepat_v1"] - 1
    return p


def _spread(g: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = {"n": len(g)}
    for c in cols:
        out[f"{c}_mean"] = g[c].mean()
        out[f"{c}_sd"] = g[c].std(ddof=1) if len(g) > 1 else np.nan
        out[f"{c}_min"] = g[c].min()
        out[f"{c}_max"] = g[c].max()
    return pd.Series(out)


def aggregate() -> None:
    files = sorted((OUT / "runs").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(OUT / "all_rows.csv", index=False)
    ratios = _ratio_rows(df)
    ratios.to_csv(OUT / "ratios.csv", index=False)

    seeds = df[df.run.str.startswith("seed_") & (df.bench_variant == "base")]
    seed_tab = seeds.groupby(["branch", "ranking"]).apply(_spread, KEY, include_groups=False).reset_index()
    seed_tab["ranking"] = pd.Categorical(seed_tab.ranking, RANKINGS, ordered=True)
    seed_tab = seed_tab.sort_values(["branch", "ranking"])
    seed_tab.to_csv(OUT / "seeds_summary.csv", index=False)
    r_seed = ratios[ratios.run.str.startswith("seed_") & (ratios.bench_variant == "base")]
    seed_ratio = r_seed.groupby("branch").apply(_spread, ["x_rules", "x_random", "v2_minus_v1", "v21_minus_v1"],
                                                include_groups=False).reset_index()
    means = r_seed.groupby("branch")[["tepat_v2_1", "rules_only", "random"]].mean()
    seed_ratio["x_rules_of_means"] = (means.tepat_v2_1 / means.rules_only).values
    seed_ratio["x_random_of_means"] = (means.tepat_v2_1 / means.random).values
    seed_ratio.to_csv(OUT / "seeds_ratios.csv", index=False)

    # Sensitivity: one dial at a time, averaged over SENS_SEEDS (default point = the seed runs for those seeds).
    base_runs = ratios.seed.isin(SENS_SEEDS) & ratios.run.str.startswith("seed_")
    sens = []

    def add(dial, value, mask):
        sub = ratios[mask]
        t2 = df.merge(sub[["run", "bench_variant", "branch"]], on=["run", "bench_variant", "branch"])
        for b in ["B", "C"]:
            s = sub[sub.branch == b]
            row = dict(dial=dial, value=value, branch=b, n_seeds=len(s),
                       v21_rp_hour=s.tepat_v2_1.mean(), rules_rp_hour=s.rules_only.mean(), random_rp_hour=s.random.mean(),
                       x_rules_of_means=s.tepat_v2_1.mean() / s.rules_only.mean(), x_rules_min=s.x_rules.min(), x_rules_max=s.x_rules.max(),
                       x_random_of_means=s.tepat_v2_1.mean() / s.random.mean(), x_random_min=s.x_random.min())
            for rk, tag in [("tepat_v2_1", "v21"), ("rules_only", "rules")]:
                tt = t2[(t2.branch == b) & (t2.ranking == rk)]
                row[f"{tag}_precision"] = tt.precision.mean()
                row[f"{tag}_honest_flag"] = tt.honest_flag_rate.mean()
            sens.append(row)

    add("default", "prev 20%, lookback 12, budget 10%, hours 3/6/12/24", base_runs & (ratios.bench_variant == "base"))
    for p in PREVALENCE:
        add("prevalence", f"{int(p * 100)}%", ratios.run.str.startswith(f"prev{int(p * 100):02d}_") & (ratios.bench_variant == "base"))
    for lb in LOOKBACK:
        add("lookback_months", str(lb), ratios.run.str.startswith(f"lb{lb:02d}_") & (ratios.bench_variant == "base"))
    for x in BUDGETS:
        add("budget_frac", f"{int(x * 100)}%", base_runs & (ratios.bench_variant == f"budget{int(x * 100):02d}"))
    for k, v in HOURS.items():
        add("inspection_hours", f"{k} {'/'.join(map(str, v))}", base_runs & (ratios.bench_variant == f"hours_{k}"))
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(OUT / "sensitivity.csv", index=False)

    # Variant recall and look-alike flag rates for TEPAT v2.1 across seeds.
    v = seeds[seeds.ranking == "tepat_v2_1"]
    rec_cols = [c for c in v.columns if c.startswith("recall_") or c.startswith("flag_") or c == "rp_range_coverage"]
    var_tab = v.groupby("branch")[rec_cols].agg(["mean", "min", "max"]).T
    var_tab.to_csv(OUT / "variant_recall.csv")
    print(seed_tab.round(3).to_string(index=False))
    print(seed_ratio.round(3).to_string(index=False))
    print(sens_df.round(3).to_string(index=False))
    print(json.dumps({"runs": len(files), "rows": len(df)}))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    r = sub.add_parser("run")
    r.add_argument("name")
    r.add_argument("--keep-data", action="store_true")
    a = sub.add_parser("all")
    a.add_argument("--jobs", type=int, default=2)
    a.add_argument("--keep-data", action="store_true")
    sub.add_parser("aggregate")
    args = ap.parse_args()
    if args.cmd == "plan":
        for k, v in plan().items():
            print(k, v)
    elif args.cmd == "run":
        run_one(args.name, args.keep_data)
    elif args.cmd == "all":
        run_all(args.jobs, args.keep_data)
    else:
        aggregate()


if __name__ == "__main__":
    main()
