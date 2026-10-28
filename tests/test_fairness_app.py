"""The fairness view renders real group denominators, not only chart rates."""
import json
from pathlib import Path

import pandas as pd
import pytest
import pyarrow.ipc as ipc
from streamlit.testing.v1 import AppTest

from tepat import bench

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
VIEW = "Keadilan & jalur acak"


@pytest.mark.parametrize("branch", ["A", "B", "C"])
def test_fairness_view_default_region_denominators_match_demo(monkeypatch, branch):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.selectbox(key="branch").set_value(branch).run()
    app.sidebar.radio(key="view").set_value(VIEW).run()
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 3, "Show sector, size-band and selected-dimension diagnostics"
    tables = [table.value for table in app.dataframe if "n_honest_selected" in table.value.columns]
    assert len(tables) == 1, "Show count denominators alongside default-seed rates"
    actual = tables[0]
    assert set(actual["group"]) == {"kabkota"}
    f = pd.read_parquet(ROOT / "demo" / branch / "features.parquet")
    t = pd.read_parquet(ROOT / "demo" / branch / "truth" / "employers_truth.parquet").set_index("employer_id")
    ranked = pd.read_parquet(ROOT / "demo" / branch / "ranked.parquet")
    rb = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))["branches"][branch]
    picked = bench.select_by_budget(list(ranked.index), f.hours, rb["budget_hours"])
    expected = bench.fairness(picked, f, t).query("group == 'kabkota'")
    for col in ["n_total", "n_selected", "n_violators", "n_honest", "n_honest_selected", "pick_rate", "violation_rate", "honest_flag_rate"]:
        pd.testing.assert_series_equal(actual.set_index("value")[col], expected.set_index("value")[col], check_names=False)
    text = "\n".join(str(e.value) for kind in ["caption", "info", "warning", "markdown"] for e in app.get(kind))
    assert "kab/kota fiktif" in text
    assert "n_honest_selected / n_honest" in text
    assert "sektor dan skala" not in text
    assert "menurut band ukuran" in text
    if branch == "A":
        assert any("bukan hasil uji tertahan" in e.value for e in app.warning)


@pytest.mark.parametrize("branch", ["A", "B", "C"])
def test_fairness_view_cross_seed_coverage_matches_real_raw_runs(monkeypatch, branch):
    from tepat.fairness_eval import SEEDS, aggregate_groups

    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.selectbox(key="branch").set_value(branch).run()
    app.sidebar.radio(key="view").set_value(VIEW).run()
    assert not app.exception
    tables = [table.value for table in app.dataframe if "pick_rate_n_seeds" in table.value.columns]
    assert len(tables) == 1, "Render real across-seed rates and coverage, not single-seed proxy ranges"
    payloads = [json.loads(path.read_text(encoding="utf-8")) for seed in SEEDS
                if (path := ROOT / "results" / "fairness" / "runs" / f"seed_{seed}.json").is_file()]
    rows = pd.DataFrame([row for payload in payloads for row in payload["groups"]])
    expected = aggregate_groups(rows).query("branch == @branch and group == 'kabkota'")
    actual = tables[0]
    for column in ["pick_rate_mean", "pick_rate_min", "pick_rate_max", "pick_rate_n_seeds", "honest_flag_rate_n_seeds", "n_seeds_expected", "n_seeds_present"]:
        pd.testing.assert_series_equal(actual.set_index("value")[column], expected.set_index("value")[column], check_names=False)
    counts = [table.value for table in app.dataframe if "n_honest_selected_sum" in table.value.columns]
    assert len(counts) == 1
    assert counts[0].n_total_sum.sum() == 1500 * len(payloads)
    assert counts[0].n_honest_sum.sum() == expected.n_honest_sum.sum()
    text = "\n".join(str(e.value) for kind in ["caption", "warning", "info", "markdown"] for e in app.get(kind))
    assert "bukan interval kepercayaan" in text
    assert "bukan kestabilan peringkat perusahaan" in text
    assert "bukan perusahaan unik" in text


@pytest.mark.parametrize("branch", ["A", "B", "C"])
def test_fairness_front_screen_uses_stored_benchmark_and_group_rates(monkeypatch, branch):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.selectbox(key="branch").set_value(branch).run()
    app.sidebar.radio(key="view").set_value(VIEW).run()
    assert not app.exception
    benchmark = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))["branches"][branch]
    arm = benchmark["rankings"][f"{benchmark['final_ranking']}_with_random_arm"]
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Anggaran jalur acak"] == "8%"
    assert metrics["Perusahaan sampel acak"] == str(arm["random_arm_size"])
    assert metrics["Pelanggaran di sampel acak"] == f"{arm['random_arm_violation_rate']:.0%}"
    assert metrics["Jujur ikut terpilih"] == f"{arm['honest_flag_rate']:.1%}".replace(".", ",")
    charts = app.get("vega_lite_chart")
    groups = pd.DataFrame(json.loads((ROOT / "results" / "fairness" / "runs" / "seed_20261007.json").read_text(encoding="utf-8"))["groups"])
    for chart, dimension in zip(charts[:2], ["sector", "band_obs"]):
        records = ipc.open_stream(chart.proto.datasets[0].data.data).read_all().to_pandas()
        assert not records.empty, f"{dimension} chart should contain saved group data"
        expected = groups.query("branch == @branch and group == @dimension")
        assert len(records) == len(expected) * 3
        assert set(records.loc[records.metric == "Terpilih", "rate"]) == set(expected.pick_rate)
        assert json.loads(chart.proto.spec)["config"]["background"] == "transparent"
    text = "\n".join(str(e.value) for kind in ["caption", "info", "warning", "markdown"] for e in app.get(kind))
    assert "bukan sektor atau wilayah" in text


@pytest.mark.parametrize("dimension", ["sector", "band_obs", "kabkota"])
def test_fairness_dimension_keeps_real_denominators_and_five_seed_tables(monkeypatch, dimension):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.sidebar.radio(key="view").set_value(VIEW).run()
    app.selectbox(key="fairness_dimension").set_value(dimension).run()
    assert not app.exception
    assert len(app.get("vega_lite_chart")) == 3
    for column in ["n_honest_selected", "pick_rate_n_seeds", "n_honest_selected_sum"]:
        tables = [table.value for table in app.dataframe if column in table.value.columns]
        assert len(tables) == 1
        assert set(tables[0].group) == {dimension}


def test_sector_gap_note_is_descriptive_and_computed_from_saved_rates(monkeypatch):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.selectbox(key="branch").set_value("B").run()
    app.sidebar.radio(key="view").set_value(VIEW).run()
    groups = pd.DataFrame(json.loads((ROOT / "results" / "fairness" / "runs" / "seed_20261007.json").read_text(encoding="utf-8"))["groups"])
    sector = groups.query("branch == 'B' and group == 'sector' and small_n == False").copy()
    row = sector.loc[(sector.violation_rate - sector.pick_rate).idxmax()]
    captions = "\n".join(c.value for c in app.caption)
    assert f"{row.violation_rate:.1%}" in captions
    assert f"{row.pick_rate:.1%}" in captions
    assert "bukan alasan mengubah bobot jalur acak" in captions


def test_panel_warns_about_undefined_rates_outside_default_seed(tmp_path):
    from tepat.fairness_eval import SEEDS

    features = pd.DataFrame({"sector": ["C", "C"], "band_obs": ["2-9", "2-9"], "kabkota": ["Fiksi", "Fiksi"]}, index=["x", "y"])
    truth = pd.DataFrame({"is_violator": [True, False]}, index=features.index)
    directory = tmp_path / "fairness"
    (directory / "runs").mkdir(parents=True)
    for seed, all_violators in [(SEEDS[0], False), (SEEDS[1], True)]:
        labels = truth.copy()
        if all_violators:
            labels["is_violator"] = True
        frame = bench.fairness(["x"], features, labels).assign(seed=seed, branch="B", ranking="tepat_v2_1")
        records = frame.astype(object).where(frame.notna(), None).to_dict("records")
        (directory / "runs" / f"seed_{seed}.json").write_text(json.dumps({"groups": records}), encoding="utf-8")
    script = f"""import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'app')!r})
import fairness_panel
fairness_panel.render('B', Path({str(directory)!r}), {{'rankings': {{}}}}, {{'C': 'Manufaktur'}})
"""
    app = AppTest.from_string(script, default_timeout=120).run()
    assert not app.exception
    assert any("tingkat lintas seed tidak terdefinisi" in w.value for w in app.warning)
    rates = [table.value for table in app.dataframe if "honest_flag_rate_n_seeds" in table.value.columns][0]
    assert rates.iloc[0].honest_flag_rate_n_seeds == 1
    assert rates.iloc[0].honest_flag_rate_undefined_seeds == 1
    assert rates.iloc[0].n_seeds_present == 2


def test_fairness_sources_and_protocol_avoid_typographic_tells():
    import re

    emoji = re.compile(r"[\U0001f300-\U0001faff\u2600-\u27bf\ufe0f]")
    for relative in ["app/fairness_panel.py", "scripts/fairness_sweep.py", "src/tepat/fairness_eval.py", "results/fairness/PROTOCOL.md"]:
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert not emoji.search(text)
        for glyph in ["\u2014", "\u2013", "\u00b7", "\u2026"]:
            assert glyph not in text, relative
