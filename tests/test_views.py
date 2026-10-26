"""The registration list view and the 5-seed Benchmark section, on the committed demo slice."""
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
DEMO = ROOT / "demo"
RESULTS = ROOT / "results"
FULL_UNREG = (ROOT / "data" / "B" / "truth" / "unregistered_truth.parquet").is_file() and \
    (RESULTS / "unreg_ranked_B.parquet").is_file()


def kali(x: float) -> str:
    return f"{x:.1f}".replace(".", ",") + "x"


def open_view(monkeypatch, branch: str, view: str) -> AppTest:
    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.selectbox(key="branch").set_value(branch).run()
    app.sidebar.radio(key="view").set_value(view).run()
    assert not app.exception
    return app


def metric(app: AppTest, label: str) -> str:
    found = [m.value for m in app.metric if m.label == label]
    assert found, f"no metric {label!r}: {[m.label for m in app.metric]}"
    return found[0]


@pytest.mark.parametrize("branch", ["B", "C"])
def test_registration_view_matches_the_saved_benchmark(monkeypatch, branch):
    app = open_view(monkeypatch, branch, "Usaha belum terdaftar")
    ranked = pd.read_parquet(DEMO / branch / "unreg_ranked.parquet")
    assert len(app.dataframe[0].value) == len(ranked) == 800
    s = pd.read_csv(RESULTS / "unregistered_summary.csv").query("branch == @branch").set_index("ranking")
    assert metric(app, "Dibanding aturan hukum") == kali(s.at["tepat_registration", "rp_h"] / s.at["rules_only", "rp_h"])
    assert metric(app, "Presisi TEPAT") == f"{s.at['tepat_registration', 'precision']:.0%}"
    captions = "\n".join(c.value for c in app.caption)
    assert "Keunggulan tipis" in captions, "A modest gain must be called modest."
    assert "Usaha tanpa NIB tetap tidak terlihat" in captions


def test_registration_case_and_answer_key_follow_the_selection(monkeypatch):
    app = open_view(monkeypatch, "B", "Usaha belum terdaftar")
    ranked = pd.read_parquet(DEMO / "B" / "unreg_ranked.parquet")
    nib = str(ranked.index[5])
    app.selectbox(key="unreg_pick").set_value(nib).run()
    assert not app.exception
    assert any(h.value == ranked.iloc[5]["name"] for h in app.subheader)


def test_registration_view_on_tuning_branch_warns_and_skips_test_numbers(monkeypatch):
    app = open_view(monkeypatch, "A", "Usaha belum terdaftar")
    assert any("Cabang A dipakai untuk melatih" in w.value for w in app.warning)
    assert not any(m.label == "Dibanding aturan hukum" for m in app.metric)


@pytest.mark.parametrize("branch", ["B", "C"])
def test_benchmark_shows_five_seed_ranges_and_no_stale_caveats(monkeypatch, branch):
    app = open_view(monkeypatch, branch, "Benchmark")
    r = pd.read_csv(RESULTS / "sweep" / "seeds_ratios.csv").set_index("branch").loc[branch]
    s = pd.read_csv(RESULTS / "sweep" / "seeds_summary.csv").query("branch == @branch and ranking == 'tepat_v2_1'").iloc[0]
    assert metric(app, "Dibanding acak") == kali(r.x_random_of_means)
    rp = metric(app, "Rp per jam, lima seed")
    assert rp == "Rp" + f"{s.rp_per_hour_juta_mean:.1f}".replace(".", ",") + " jt"
    text = "\n".join([c.value for c in app.caption] + [w.value for w in app.warning])
    for stale in ("Satu seed", "satu seed.", "belum ada interval", "tidak pernah mendaftar sama sekali tidak terlihat"):
        assert stale not in text, f"stale caveat still shown: {stale!r}"
    assert "Usaha belum terdaftar" in text, "The limit should point to where unregistered businesses are handled."


def test_headline_numbers_on_the_benchmark_match_the_proposal():
    """Slides quote B: Rp8,9 jt per hour (7,1 s.d. 11,1), 2,3x rules (1,9 s.d. 3,7), 8x random."""
    r = pd.read_csv(RESULTS / "sweep" / "seeds_ratios.csv").set_index("branch").loc["B"]
    s = pd.read_csv(RESULTS / "sweep" / "seeds_summary.csv").query("branch == 'B' and ranking == 'tepat_v2_1'").iloc[0]
    assert (round(s.rp_per_hour_juta_mean, 1), round(s.rp_per_hour_juta_min, 1), round(s.rp_per_hour_juta_max, 1)) == (8.9, 7.1, 11.1)
    assert (kali(r.x_rules_of_means), kali(r.x_rules_min), kali(r.x_rules_max)) == ("2,3x", "1,9x", "3,7x")
    assert round(r.x_random_of_means) == 8


@pytest.mark.skipif(not FULL_UNREG, reason="needs the full local run (uv run tepat all && uv run tepat unreg)")
@pytest.mark.parametrize("branch", ["A", "B", "C"])
def test_registration_slice_is_the_full_run(branch):
    pd.testing.assert_frame_equal(pd.read_parquet(RESULTS / f"unreg_ranked_{branch}.parquet"),
                                  pd.read_parquet(DEMO / branch / "unreg_ranked.parquet"))
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT / "data" / branch / "truth" / "unregistered_truth.parquet"),
                                  pd.read_parquet(DEMO / branch / "truth" / "unregistered_truth.parquet"))
    manifest = json.loads((DEMO / "manifest.json").read_text(encoding="utf-8"))
    assert f"{branch}/unreg_ranked.parquet" in manifest["files"]
