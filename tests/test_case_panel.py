"""Case panel data and real Streamlit interactions."""
from pathlib import Path
import sys

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))


def test_case_composition_uses_observed_feature_values():
    from case_panel import composition

    features = pd.read_parquet(ROOT / "demo/B/features.parquet")
    row = features.iloc[0]
    frame = composition(row, 12)
    assert (frame.Rp > 0).all()
    assert frame.loc[frame.Komponen == "Pekerja tidak terdaftar", "Rp"].sum() == row.rp_headcount * 12
    assert frame.loc[frame.Komponen == "Biaya rawat saat nonaktif", "Rp"].sum() == max(0, row.rp_care_liability)


def test_case_panel_preserves_report_and_follow_up_actions(tmp_path):
    script = tmp_path / "case_app.py"
    script.write_text('''
from pathlib import Path
import sys
import pandas as pd
import streamlit as st
from tepat import cases, reports
sys.path.insert(0, r"''' + str(ROOT / "app") + '''")
import case_panel
root = Path(r"''' + str(ROOT) + '''")
branch_dir = root / "demo/B"
ranked = pd.read_parquet(branch_dir / "ranked.parquet")
F = pd.read_parquet(branch_dir / "features.parquet")
reg = pd.read_parquet(branch_dir / "observed/employer_registry.parquet").set_index("employer_id")
obs = {"billing": pd.read_parquet(branch_dir / "observed/billing.parquet")}
if "case_store" not in st.session_state:
    st.session_state.case_store = cases.CaseStore.sandbox()
store = st.session_state.case_store
if "reports" not in st.session_state:
    st.session_state.reports = reports.ReportStore(store)
eid = st.selectbox("Pilih perusahaan", list(ranked.index), key="case_employer")
case_panel.render(branch="B", eid=eid, ranked=ranked, F=F, reg=reg, obs=obs,
    store=store, reports=st.session_state.reports, branch_dir=branch_dir,
    sektor={"C":"Manufaktur", "F":"Konstruksi", "G":"Perdagangan", "H":"Logistik", "I":"Akomodasi", "J":"Informasi", "K":"Keuangan", "Q":"Kesehatan", "A":"Pertanian"},
    chart_config={"view": {"stroke": None}}, open_case=lambda b,e: st.session_state.update(opened=(b,e)),
    open_inbox=lambda b,e: st.session_state.update(inbox=(b,e)),
    load_truth=lambda p: pd.read_parquet(Path(p) / "truth/employers_truth.parquet").set_index("employer_id"),
    jt=lambda x: f"Rp{x/1e6:.1f} jt", rp=lambda x: f"Rp{x:.0f}",
    tanggal=lambda x: str(x), waktu=lambda x: str(x),
    flash=lambda x: st.session_state.update(fu_flash=x), lookback=12)
''', encoding="utf-8")
    app = AppTest.from_file(str(script), default_timeout=120).run()
    assert not app.exception
    assert [m.label for m in app.metric] == ["Perkiraan pemulihan", "Peringkat", "P(pelanggaran)", "Waktu pemeriksaan"]
    assert len([w for w in app.selectbox if w.label == "Pilih perusahaan"]) == 1
    assert app.button(key="case_reports")
    app.button(key="case_reports").click().run()
    assert app.session_state["inbox"] == ("B", app.selectbox(key="case_employer").value)
    assert app.button(key="draft_generate")
    store = app.session_state["case_store"]
    ranked = pd.read_parquet(ROOT / "demo/B/ranked.parquet")
    eid = next(e for e in ranked[ranked.queue == "inspeksi"].index if store.get("B", e) is None)
    app.selectbox(key="case_employer").set_value(eid).run()
    assert not app.exception
    app.button(key="fu_start").click().run()
    assert not app.exception
    assert store.get("B", eid)["status"] == "Baru"
    app.radio(key="draft_type").set_value("surat_kepatuhan").run()
    app.button(key="draft_generate").click().run()
    app.button(key="draft_approve").click().run()
    assert not app.exception
    case = store.get("B", eid)
    assert store.events(case["case_id"]).kind.iloc[-1] == "draf disetujui"
    assert case["status"] == "Baru"
    app.button(key="fu_apply_suggest").click().run()
    assert store.get("B", eid)["status"] == "Surat dikirim"
    app.radio(key=f"fu_outcome_B_{case['case_id']}_{len(store.events(case['case_id']))}").set_value("koreksi_mandiri")
    [b for b in app.button if b.label == "Tutup kasus"][0].click().run()
    assert not app.exception
    assert store.get("B", eid)["status"] == "Selesai"
    app.text_input(key=f"fu_reopen_reason_B_{case['case_id']}_{len(store.events(case['case_id']))}").input("Bukti baru diterima")
    [b for b in app.button if b.label == "Buka kembali"][0].click().run()
    assert not app.exception
    assert store.get("B", eid)["status"] == "Menunggu koreksi"


def test_case_panel_has_no_banned_punctuation():
    source = (ROOT / "app/case_panel.py").read_text(encoding="utf-8")
    assert not any(g in source for g in ("\u2014", "\u2013", "\u00b7", "\u2026"))
