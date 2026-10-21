"""Fairness is an evaluation-only diagnostic over the observed employer universe."""
import pandas as pd
import pytest

from tepat import bench


def population():
    features = pd.DataFrame({
        "sector": ["C", "C", "G", "G"],
        "band_obs": ["2-9", "2-9", "10-49", "10-49"],
        "kabkota": ["Kab. Arunika", "Kab. Arunika", "Kota Candrapura", "Kota Candrapura"],
    }, index=pd.Index(["e1", "e2", "e3", "e4"], name="employer_id"))
    truth = pd.DataFrame({"is_violator": [True, False, False, True]}, index=features.index)
    return features, truth


def test_regional_breakdown_has_explicit_denominators():
    features, truth = population()
    rows = bench.fairness(["e1", "e2", "e3"], features, truth)
    assert set(rows.group) == {"sector", "band_obs", "kabkota"}
    region = rows.set_index(["group", "value"]).loc[("kabkota", "Kab. Arunika")]
    assert region[["n_total", "n_selected", "n_violators", "n_honest", "n_honest_selected"]].tolist() == [2, 2, 1, 1, 1]
    assert region.pick_rate == 1
    assert region.violation_rate == 0.5
    assert region.honest_flag_rate == 1
    assert region.small_n
    assert "small_n_total" in region.warning
    for group in ("sector", "band_obs", "kabkota"):
        part = rows[rows.group == group]
        assert part.n_total.sum() == 4
        assert part.n_selected.sum() == 3
        assert part.n_honest_selected.sum() == 2


def test_missing_region_and_empty_groups_are_not_measured_zero():
    features, truth = population()
    features.loc[["e3", "e4"], "kabkota"] = None
    features["sector"] = pd.Categorical(features.sector, categories=["C", "G", "Q"])
    truth.loc[["e1", "e2"], "is_violator"] = True
    rows = bench.fairness(["e1"], features, truth).set_index(["group", "value"])
    missing = rows.loc[("kabkota", "__missing__")]
    assert missing.n_total == 2
    assert "missing_label" in missing.warning
    assert missing.pick_rate == 0  # A measured zero: two known unselected employers.
    all_violators = rows.loc[("sector", "C")]
    assert all_violators.n_honest == 0
    assert pd.isna(all_violators.honest_flag_rate)
    assert "no_honest" in all_violators.warning
    empty = rows.loc[("sector", "Q")]
    assert empty.n_total == 0
    assert all(pd.isna(empty[k]) for k in ("pick_rate", "violation_rate", "honest_flag_rate"))
    assert "empty_group" in empty.warning


@pytest.mark.parametrize("bad_input", ["duplicate_features", "duplicate_truth", "missing_truth", "extra_truth", "duplicate_pick", "unknown_pick", "null_truth", "numeric_truth", "null_key"])
def test_invalid_entity_alignment_is_rejected(bad_input):
    features, truth = population()
    picked = ["e1"]
    if bad_input == "duplicate_features":
        features = pd.concat([features, features.iloc[:1]])
    elif bad_input == "duplicate_truth":
        truth = pd.concat([truth, truth.iloc[:1]])
    elif bad_input == "missing_truth":
        truth = truth.iloc[:-1]
    elif bad_input == "extra_truth":
        truth.loc["extra"] = False
    elif bad_input == "duplicate_pick":
        picked = ["e1", "e1"]
    elif bad_input == "unknown_pick":
        picked = ["extra"]
    elif bad_input == "null_truth":
        truth["is_violator"] = truth.is_violator.astype("boolean")
        truth.loc["e2", "is_violator"] = pd.NA
    elif bad_input == "numeric_truth":
        truth["is_violator"] = truth.is_violator.astype(int)
    elif bad_input == "null_key":
        features.index = ["e1", "e2", "e3", None]
    with pytest.raises(ValueError):
        bench.fairness(picked, features, truth)


def test_registry_region_fallback_and_explicit_group_universe():
    features, truth = population()
    registry = features[["kabkota"]].iloc[::-1]
    features = features.drop(columns="kabkota")
    rows = bench.fairness([], features, truth, registry=registry,
                          group_values={"kabkota": ["Kab. Arunika", "Kota Candrapura", "Kab. Kosong"]})
    assert rows.query("group == 'kabkota'").n_total.sum() == 4
    assert rows.query("value == 'Kab. Kosong'").iloc[0].n_total == 0
    with pytest.raises(ValueError, match="registry"):
        bench.fairness([], features, truth, registry=pd.concat([registry, registry.iloc[:1]]))
    missing = bench.fairness([], features, truth)
    assert missing.query("group == 'kabkota'").iloc[0].value == "__missing__"
    empty = bench.fairness([], features.iloc[:0], truth.iloc[:0], group_values={"kabkota": ["Kab. Kosong"]})
    assert empty.iloc[0].n_total == 0
    assert pd.isna(empty.iloc[0].pick_rate)
