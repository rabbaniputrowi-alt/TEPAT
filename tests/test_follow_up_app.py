"""Follow-up flow through the real app: start, assign, close with an outcome, bulk start, board."""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def _app():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def _submit(app, label):
    [b for b in app.button if b.label == label][0].click().run()
    assert not app.exception


def _w(app, kind, label):
    return [w for w in app.get(kind) if w.label == label][0]


def _text(app):
    parts = []
    for kind in ("markdown", "caption", "success", "error", "info"):
        parts += [str(getattr(e, "value", "")) for e in app.get(kind)]
    return "\n".join(parts)


def test_board_starts_with_private_examples_at_every_stage():
    app = _app()
    app.sidebar.radio(key="view").set_value("Tindak lanjut").run()
    assert not app.exception
    store = app.session_state["case_store"]
    assert store.mode == "sandbox"
    cases = store.cases("B")
    assert len(cases) >= 8 and set(cases.source) == {"contoh"}
    assert "Versi daring" in _text(app)
    board = app.dataframe[0].value
    assert set(board.Status) >= {"Baru", "Ditugaskan", "Selesai"}
    assert app.get("download_button")


def test_case_can_be_started_assigned_and_closed_only_with_an_outcome():
    app = _app()
    store = app.session_state["case_store"]
    app.sidebar.radio(key="view").set_value("Kasus perusahaan").run()
    ranked = pd.read_parquet(Path(__file__).resolve().parents[1] / "demo" / "B" / "ranked.parquet")
    eid = next(e for e in ranked.sort_values("rank").index if store.get("B", e) is None)
    app.selectbox(key="case_employer").set_value(eid).run()
    assert not app.exception
    app.button(key="fu_start").click().run()
    assert not app.exception
    assert store.get("B", eid)["status"] == "Baru"
    _w(app, "selectbox", "Pemeriksa").set_value("Pemeriksa B")
    _w(app, "text_area", "Catatan").input("Telepon personalia")
    _submit(app, "Simpan perubahan")
    case = store.get("B", eid)
    assert (case["status"], case["owner"]) == ("Ditugaskan", "Pemeriksa B")
    # the form shows the saved case, so a second save cannot silently undo the first
    assert _w(app, "selectbox", "Status").value == "Ditugaskan"
    assert _w(app, "text_area", "Catatan").value == ""
    _submit(app, "Simpan perubahan")
    assert store.get("B", eid)["status"] == "Ditugaskan"
    _submit(app, "Tutup kasus")  # no outcome chosen
    assert store.get("B", eid)["status"] == "Ditugaskan"
    assert "Pilih hasil" in _text(app)
    _w(app, "radio", "Hasil pemeriksaan").set_value("koreksi_mandiri")
    _w(app, "number_input", "Rp dipulihkan (0 bila belum ada)").set_value(4_000_000)
    _submit(app, "Tutup kasus")
    case = store.get("B", eid)
    assert (case["status"], case["outcome"], case["recovered_rp"]) == ("Selesai", "koreksi_mandiri", 4_000_000)
    kinds = store.events(case["case_id"]).kind.tolist()
    assert kinds == ["dibuka", "status", "pemeriksa", "catatan", "selesai"]


def test_bulk_start_covers_the_budget_and_skips_existing_cases():
    app = _app()
    store = app.session_state["case_store"]
    before = store.count("B")
    app.button(key="bulk_follow_up").click().run()
    assert not app.exception
    created = store.count("B") - before
    assert created > 50
    assert app.button(key="bulk_follow_up").disabled  # nothing left to start
    assert set(store.cases("B").source) == {"contoh", "massal"}


def test_approved_draft_lands_in_case_history_and_offers_next_status():
    app = _app()
    store = app.session_state["case_store"]
    app.sidebar.radio(key="view").set_value("Kasus perusahaan").run()
    ranked = pd.read_parquet(Path(__file__).resolve().parents[1] / "demo" / "B" / "ranked.parquet")
    eid = next(e for e in ranked[ranked.queue == "inspeksi"].sort_values("rank").index if store.get("B", e) is None)
    app.selectbox(key="case_employer").set_value(eid).run()
    # approving without a case only explains how to record it
    app.radio(key="draft_type").set_value("surat_kepatuhan").run()
    app.button(key="draft_generate").click().run()
    app.button(key="draft_approve").click().run()
    assert not app.exception
    assert store.get("B", eid) is None
    assert "belum tercatat di riwayat" in _text(app)
    app.button(key="fu_start").click().run()
    app.button(key="draft_generate").click().run()
    app.button(key="draft_approve").click().run()
    assert not app.exception
    case = store.get("B", eid)
    ev = store.events(case["case_id"])
    assert ev.kind.iloc[-1] == "draf disetujui" and "Tidak dikirim" in ev.detail.iloc[-1]
    assert case["status"] == "Baru"  # approval alone never changes the status
    app.button(key="fu_apply_suggest").click().run()
    assert not app.exception
    assert store.get("B", eid)["status"] == "Surat dikirim"
