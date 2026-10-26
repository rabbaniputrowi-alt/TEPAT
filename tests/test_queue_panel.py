"""Daftar prioritas after the 3a redesign: holdback math, curve data, number formats, EN labels, card selection."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))
import queue_panel as Q  # noqa: E402


def _app(monkeypatch) -> AppTest:
    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    return app


def _markup(app: AppTest) -> str:
    return "\n".join(str(e.value) for e in app.markdown)


# ---------------------------------------------------------------- pure helpers

def test_holdback_keeps_eight_percent_of_hours_out_of_the_queue():
    hours = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=list("abcde"))
    picked = Q.select_in_budget(hours, 100)  # 92 effective hours: 10 + 20 + 30 = 60 fits, + 40 = 100 does not
    assert list(picked) == ["a", "b", "c"]
    assert Q.effective_hours(1000) == pytest.approx(920)
    assert list(Q.select_in_budget(hours, 100, holdback=0)) == ["a", "b", "c", "d"]


def test_recovery_curves_start_at_zero_rise_monotonically_and_share_one_value_measure():
    hours = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    value = pd.Series([40.0, 30.0, 20.0, 10.0], index=list("abcd"))
    c = Q.recovery_curves(pd.Index(list("abcd")), pd.Index(list("dcba")), hours, value, xmax=100, labels=("T", "R"))
    for label in ("T", "R"):
        part = c[c.metode == label]
        assert part.iloc[0][["jam", "rp"]].tolist() == [0.0, 0.0]
        assert part.jam.is_monotonic_increasing and part.rp.is_monotonic_increasing
        assert part.rp.iloc[-1] == pytest.approx(value.sum())  # same measure, same total
    t, r = c[c.metode == "T"], c[c.metode == "R"]
    assert t[t.jam == 3].rp.item() == 70 and r[r.jam == 4].rp.item() == 10  # each order sums the same values its own way
    assert Q.recovery_curves(pd.Index(list("abcd")), pd.Index(list("dcba")), hours, value, xmax=3).jam.max() == 3
    assert Q.value_at(pd.Index(list("abcd")), hours, value, 3.5) == (70.0, 2)


def test_numbers_follow_the_language():
    assert Q.money(7_436_100_000, "ID") == "Rp7.436,1 jt"
    assert Q.money(7_436_100_000, "EN") == "Rp7,436.1M"
    assert Q.num(1500, "ID") == "1.500" and Q.num(1500, "EN") == "1,500"
    assert Q.ratio(2.28, "ID") == "2,3x" and Q.ratio(2.28, "EN") == "2.3x"


def test_both_languages_carry_the_same_labels_and_no_typographic_tells():
    assert set(Q.T["ID"]) == set(Q.T["EN"])
    source = (ROOT / "app" / "queue_panel.py").read_text(encoding="utf-8")
    for glyph in ("\u2014", "\u2013", "\u00b7", "\u2026"):
        assert glyph not in source


# ---------------------------------------------------------------- the real app on the demo slice

def test_kpis_use_the_holdback_and_badges_come_from_the_saved_benchmark(monkeypatch):
    app = _app(monkeypatch)
    ranked = pd.read_parquet(ROOT / "demo" / "B" / "ranked.parquet")
    F = pd.read_parquet(ROOT / "demo" / "B" / "features.parquet")
    insp = ranked[ranked.queue == "inspeksi"]
    budget = app.slider(key="budget").value
    picked = Q.select_in_budget(F.hours.loc[insp.index], budget)
    markup = _markup(app)
    assert f'<div class="kpi-value">{Q.num(len(picked), "ID")}</div>' in markup
    assert f'<div class="kpi-value">{Q.money(insp.loc[picked].expected_rp.sum(), "ID")}</div>' in markup
    rb = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))["branches"]["B"]
    best, base = rb["rankings"][rb.get("final_ranking", "tepat_v2_1")], rb["rankings"]["rules_only"]
    assert Q.ratio(best["rp_per_hour"] / base["rp_per_hour"], "ID") in markup
    assert f"{best['precision']:.0%}" in markup
    assert "benchmark, anggaran 10%" in markup, "The badges must say they come from the benchmark budget."
    assert "Jalur acak 8%" in "\n".join(c.value for c in app.caption)
    assert "Kunci jawaban:" in markup, "The curve values come from the answer key and must say so."


def test_preset_sets_the_budget_and_the_card_count(monkeypatch):
    app = _app(monkeypatch)
    F = pd.read_parquet(ROOT / "demo" / "B" / "features.parquet")
    v = Q.preset_hours(float(F.hours.sum()))[0]
    app.button(key=f"preset_{v}").click().run()
    assert not app.exception
    assert app.slider(key="budget").value == v


def test_english_switch_changes_this_view_only(monkeypatch):
    app = _app(monkeypatch)
    app.radio(key="queue_lang").set_value("EN").run()
    assert not app.exception
    assert app.title[0].value == "Priority queue"
    markup = _markup(app)
    assert "Estimated recovery" in markup and "Perkiraan pemulihan" not in markup
    assert "Rp" in markup and "M</div>" in markup
    app.sidebar.radio(key="view").set_value("Benchmark").run()
    assert app.title[0].value == "Benchmark"
    assert app.selectbox(key="branch").label == "Cabang data sintetis"


def test_language_switch_keeps_branch_and_route(monkeypatch):
    app = _app(monkeypatch)
    app.selectbox(key="branch").set_value("C").run()
    app.radio(key="queue").set_value("cicilan").run()
    app.radio(key="queue_lang").set_value("EN").run()
    assert not app.exception
    assert app.selectbox(key="branch_en").value == "C" and app.radio(key="queue_en").value == "cicilan"
    app.radio(key="queue_lang").set_value("ID").run()
    assert app.selectbox(key="branch").value == "C" and app.radio(key="queue").value == "cicilan"
    app.radio(key="queue_lang").set_value("EN").run()
    app.sidebar.radio(key="view").set_value("Benchmark").run()
    assert app.selectbox(key="branch").value == "C", "Leaving the English queue must keep the chosen branch."


def test_card_selects_the_firm_and_open_case_follows_it(monkeypatch):
    app = _app(monkeypatch)
    picks = [b.key for b in app.button if b.key and b.key.startswith("qpick_")]
    assert len(picks) == Q.CARDS
    target = picks[3].removeprefix("qpick_")
    app.button(key=picks[3]).click().run()
    assert not app.exception
    assert app.session_state["queue_sel"] == ("B", target)
    app.button(key="open_case").click().run()
    assert app.sidebar.radio(key="view").value == "Kasus perusahaan"
    assert app.selectbox(key="case_employer").value == target


def test_tier_chips_filter_cards_and_branch_change_resets_the_selection(monkeypatch):
    app = _app(monkeypatch)
    ranked = pd.read_parquet(ROOT / "demo" / "B" / "ranked.parquet")
    app.radio(key="queue_tier").set_value("Sedang").run()
    picks = [b.key.removeprefix("qpick_") for b in app.button if b.key and b.key.startswith("qpick_")]
    assert picks and set(ranked.loc[picks, "tier"]) == {"Sedang"}
    app.button(key=f"qpick_{picks[2]}").click().run()
    app.selectbox(key="branch").set_value("C").run()
    assert not app.exception
    rc = pd.read_parquet(ROOT / "demo" / "C" / "ranked.parquet")
    first = [b.key.removeprefix("qpick_") for b in app.button if b.key and b.key.startswith("qpick_")][0]
    assert first in rc.index
    app.button(key="open_case").click().run()
    assert app.selectbox(key="case_employer").value == first, "A selection from branch B must not leak into C."


def test_cicilan_route_lists_installment_firms(monkeypatch):
    app = _app(monkeypatch)
    ranked = pd.read_parquet(ROOT / "demo" / "B" / "ranked.parquet")
    app.radio(key="queue").set_value("cicilan").run()
    assert not app.exception
    picks = [b.key.removeprefix("qpick_") for b in app.button if b.key and b.key.startswith("qpick_")]
    assert picks and set(ranked.loc[picks, "queue"]) == {"cicilan"}
