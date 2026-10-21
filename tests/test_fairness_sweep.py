"""The fairness run must reproduce the existing default sweep, not a new policy."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sweep
import fairness_sweep


def test_seed_run_matches_original_sweep_base_and_resumes(tmp_path, monkeypatch):
    cfg = sweep.load_config(ROOT / "config" / "default.yaml")
    cfg["n_employers"] = 120
    monkeypatch.setattr(sweep, "load_config", lambda _: cfg)
    monkeypatch.setattr(sweep, "OUT", tmp_path / "original")
    monkeypatch.setattr(sweep, "DATA", tmp_path / "original_data")
    reference = pd.read_csv(sweep.run_one("seed_20261007"))
    path = fairness_sweep.run_seed(20261007, out_dir=tmp_path / "fairness", data_root=tmp_path / "fairness_data", cfg=cfg)
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = reference[reference.bench_variant == "base"].sort_values(["branch", "ranking"]).reset_index(drop=True)
    actual = pd.DataFrame(result["headlines"]).sort_values(["branch", "ranking"]).reset_index(drop=True)
    for column in ("inspected", "hits", "precision", "recall", "rp_per_hour_juta", "rp_share", "honest_flag_rate", "budget_hours"):
        np.testing.assert_allclose(actual[column], expected[column], rtol=1e-12, atol=1e-12)
    groups = pd.DataFrame(result["groups"])
    for branch in "ABC":
        for group in ("sector", "band_obs", "kabkota"):
            part = groups[(groups.branch == branch) & (groups.group == group)]
            assert part.n_total.sum() == 120
            assert part.n_selected.sum() == actual.query("branch == @branch and ranking == 'tepat_v2_1'").iloc[0].inspected
    assert result["metadata"]["training_random_state"] == 20261007
    assert result["metadata"]["branch_roles"] == {"A": "tuning_only", "B": "held_out", "C": "shifted"}
    assert not (tmp_path / "fairness_data" / "seed_20261007").exists()
    before = path.stat().st_mtime_ns
    monkeypatch.setattr(fairness_sweep, "generate_branch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must resume")))
    assert fairness_sweep.run_seed(20261007, out_dir=tmp_path / "fairness", data_root=tmp_path / "fairness_data", cfg=cfg) == path
    assert path.stat().st_mtime_ns == before
    fairness_sweep.aggregate(out_dir=tmp_path / "fairness", reference_dir=ROOT / "results")
    raw = pd.read_csv(tmp_path / "fairness" / "group_rows.csv")
    summary = pd.read_csv(tmp_path / "fairness" / "group_summary.csv")
    assert len(raw) == len(groups)
    assert summary.n_seeds_present.eq(1).all() and summary.n_seeds_expected.eq(5).all()
    protocol = json.loads((tmp_path / "fairness" / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["completed_seeds"] == [20261007]
    assert protocol["missing_seeds"] == sweep.SEEDS[1:]
    assert not protocol["complete"]


def test_reproduction_check_compares_committed_aggregate_and_default_seed():
    rows = pd.concat([pd.read_csv(ROOT / "results" / "sweep" / "runs" / f"seed_{seed}.csv")
                      for seed in sweep.SEEDS], ignore_index=True)
    rows = rows[rows.bench_variant == "base"]
    checks = fairness_sweep.verify_headlines(rows, reference_dir=ROOT / "results")
    assert checks.passed.all()
    assert set(checks.reference) == {"seeds_summary.csv", "results.json", "seed_runs"}
    broken = rows.copy()
    broken.loc[(broken.branch == "B") & (broken.ranking == "tepat_v2_1"), "precision"] = 0
    assert not fairness_sweep.verify_headlines(broken, reference_dir=ROOT / "results").passed.all()
    missing = fairness_sweep.verify_headlines(rows[rows.seed != sweep.SEEDS[-1]], reference_dir=ROOT / "results")
    assert not missing.passed.all()
    assert "seed_coverage" in set(missing.metric)


def test_cli_lists_real_seed_plan_and_rejects_excess_workers():
    command = [sys.executable, str(ROOT / "scripts" / "fairness_sweep.py")]
    plan = subprocess.run(command + ["plan"], capture_output=True, text=True, check=True)
    assert [int(line) for line in plan.stdout.splitlines()] == sweep.SEEDS
    invalid = subprocess.run(command + ["all", "--jobs", "3"], capture_output=True, text=True)
    assert invalid.returncode != 0
    assert "invalid choice" in invalid.stderr


def test_seed_worker_uses_existing_sweep_power_setup_before_generation(tmp_path, monkeypatch):
    def power_setup():
        raise RuntimeError("power setup reached")

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("generation started before power setup")

    monkeypatch.setattr(sweep, "_no_throttle", power_setup)
    monkeypatch.setattr(fairness_sweep, "generate_branch", unexpected_generation)
    with pytest.raises(RuntimeError, match="power setup reached"):
        fairness_sweep.run_seed(20261007, out_dir=tmp_path, data_root=tmp_path / "data")


def test_committed_five_seed_artifacts_are_complete_and_reproducible():
    from tepat.fairness_eval import aggregate_groups

    directory = ROOT / "results" / "fairness"
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["complete"]
    assert protocol["completed_seeds"] == sweep.SEEDS
    assert protocol["missing_seeds"] == []
    payloads = [json.loads((directory / "runs" / f"seed_{seed}.json").read_text(encoding="utf-8")) for seed in sweep.SEEDS]
    assert all(p["metadata"]["config"]["n_employers"] == 1500 for p in payloads)
    rows = pd.DataFrame([row for p in payloads for row in p["groups"]])
    headlines = pd.DataFrame([row for p in payloads for row in p["headlines"]])
    assert len(rows) == protocol["group_rows"] == 240
    assert len(headlines) == protocol["headline_rows"] == 105
    actual = pd.read_csv(directory / "group_summary.csv").fillna("")
    expected = aggregate_groups(rows).fillna("")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
    assert expected.n_seeds_present.eq(5).all()
    checks = fairness_sweep.verify_headlines(headlines)
    assert len(checks) == protocol["reproduction_checks"] == 1260
    assert checks.passed.all()
    # The pre-existing sweep and main default results are never rewritten.
    assert set(checks.reference) == {"seeds_summary.csv", "seed_runs", "results.json"}


def test_aggregate_does_not_mark_missing_group_as_complete(tmp_path):
    import shutil

    directory = tmp_path / "fairness"
    shutil.copytree(ROOT / "results" / "fairness" / "runs", directory / "runs")
    path = directory / "runs" / "seed_20261104.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["groups"] = [r for r in payload["groups"]
                         if not (r["branch"] == "B" and r["group"] == "kabkota" and r["value"] == "Kab. Arunika")]
    path.write_text(json.dumps(payload), encoding="utf-8")
    protocol = fairness_sweep.aggregate(out_dir=directory)
    assert not protocol["complete"], "All five headline seeds do not prove group coverage"
    assert not protocol["group_coverage_complete"]
    assert protocol["missing_group_rows"] == [{"seed": 20261104, "branch": "B", "group": "kabkota", "value": "Kab. Arunika"}]
    row = pd.read_csv(directory / "group_summary.csv").query("branch == 'B' and group == 'kabkota' and value == 'Kab. Arunika'").iloc[0]
    assert row.n_seeds_present == 4 and row.pick_rate_n_seeds == 4
