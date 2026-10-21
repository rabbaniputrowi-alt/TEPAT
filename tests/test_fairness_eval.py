"""Across-seed summaries keep missing groups distinct from measured zero."""
import pandas as pd
import pytest

from tepat import bench
from tepat import fairness_eval


def test_aggregate_reports_seed_coverage_and_counts_not_mean_denominators():
    f = pd.DataFrame({"sector": ["C", "C"], "band_obs": ["2-9", "2-9"], "kabkota": ["R", "R"]}, index=["x", "y"])
    t = pd.DataFrame({"is_violator": [True, False]}, index=f.index)
    one = bench.fairness(["x", "y"], f, t).assign(seed=1, branch="B", ranking="tepat_v2_1")
    two = bench.fairness([], f, t).assign(seed=2, branch="B", ranking="tepat_v2_1")
    # A missing regional row is NOT a measured empty or zero-rate group.
    two = two[two.group != "kabkota"]
    result = fairness_eval.aggregate_groups(pd.concat([one, two]), expected_seeds=[1, 2, 3])
    region = result.query("group == 'kabkota'").iloc[0]
    assert region.n_seeds_present == 1 and region.n_seeds_expected == 3
    assert region.seeds_missing == "2;3"
    assert region.pick_rate_mean == 1
    assert region.n_total_sum == 2 and region.n_total_min == 2 and region.n_total_max == 2
    sector = result.query("group == 'sector'").iloc[0]
    assert sector.n_total_sum == 4
    assert sector.n_selected_sum == 2
    assert sector.pick_rate_mean == 0.5
    assert sector.pick_rate_min == 0 and sector.pick_rate_max == 1
    assert sector.pick_rate_n_seeds == 2
    assert "n_total_mean" not in result


def test_aggregate_keeps_empty_and_undefined_seeds_separate():
    f = pd.DataFrame({"sector": ["C"], "band_obs": ["2-9"], "kabkota": ["R"]}, index=["x"])
    t = pd.DataFrame({"is_violator": [True]}, index=f.index)
    one = bench.fairness(["x"], f, t).assign(seed=1, branch="C", ranking="tepat_v2_1")
    two = bench.fairness([], f.iloc[:0], t.iloc[:0], group_values={"kabkota": ["R"]}).assign(seed=2, branch="C", ranking="tepat_v2_1")
    rows = pd.concat([one, two])
    result = fairness_eval.aggregate_groups(rows, expected_seeds=[1, 2, 3])
    r = result.query("group == 'kabkota'").iloc[0]
    assert r.n_seeds_present == 2 and r.n_nonempty_seeds == 1
    assert r.pick_rate_n_seeds == 1 and r.pick_rate_undefined_seeds == 1
    assert r.honest_flag_rate_n_seeds == 0 and r.honest_flag_rate_undefined_seeds == 2
    assert pd.isna(r.honest_flag_rate_mean) and pd.isna(r.honest_flag_rate_pooled)
    assert r.n_total_sum == 1 and r.n_total_min == 0
    assert r.branch_role == "shifted"
    with pytest.raises(ValueError, match="duplicate"):
        fairness_eval.aggregate_groups(pd.concat([rows, one]), expected_seeds=[1, 2, 3])
    with pytest.raises(ValueError, match="unexpected seed"):
        fairness_eval.aggregate_groups(rows, expected_seeds=[1])
