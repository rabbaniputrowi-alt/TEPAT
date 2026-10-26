"""Real Streamlit worker reports and inspector triage, using private synthetic sandboxes."""
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"


def app():
    return AppTest.from_file(str(APP), default_timeout=120).run()


def widget(at, kind, label):
    return next(w for w in at.get(kind) if w.label == label)


def click(at, label):
    widget(at, "button", label).click().run()
    assert not at.exception


def text(at):
    return "\n".join(str(e.value) for kind in ("markdown", "caption", "info", "warning", "error", "success")
                     for e in at.get(kind))


def test_worker_report_is_branch_scoped_duplicate_acknowledged_and_legacy_csv_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    monkeypatch.setenv("TEPAT_RESULTS", str(tmp_path))
    legacy = tmp_path / "tips.csv"
    old = b"waktu,tiket,nik_hash,employer_id,jenis\n2025-01-01,TPT-OLD,old,PK1,tidak_didaftarkan\n"
    legacy.write_bytes(old)
    at = app()
    assert not at.exception
    assert any(o.endswith(": Laporan pekerja") for o in at.sidebar.radio(key="view").options), "Inspectors need a real report inbox."
    cases = at.session_state["case_store"]
    reports = at.session_state["report_store"]
    before, case_count = reports.count("B"), cases.count("B")
    at.sidebar.radio(key="view").set_value("Cek kepesertaan saya").run()
    click(at, "Kirim laporan demo")
    assert reports.count("B") == before + 1
    row = reports.reports("B").query("source == 'pekerja'").iloc[0]
    ticket, employer = row.ticket, row.employer_id
    assert ticket in text(at) and "baru" in text(at).lower()
    assert "hash bukan anonimisasi produksi" in text(at).lower()
    assert "sesi" in text(at).lower() and "ulang" in text(at).lower()
    click(at, "Kirim laporan demo")
    assert reports.count("B") == before + 1 and reports.get("B", ticket)["duplicate_count"] == 1
    assert ticket in text(at) and "duplikat" in text(at).lower()
    assert cases.count("B") == case_count
    assert legacy.read_bytes() == old
    assert not (tmp_path / "cases.sqlite").exists()

    at.sidebar.radio(key="view").set_value("Laporan pekerja").run()
    assert not at.exception
    assert "tidak diimpor" in text(at) and "tidak memiliki cabang" in text(at)
    widget(at, "selectbox", "Pilih tiket").set_value(ticket).run()
    assert "dugaan" in text(at).lower() and "peringkat" in text(at).lower()
    table = at.dataframe[0].value
    assert ticket in set(table.Tiket) and "NIK" not in table.columns
    click(at, "Buka perusahaan")
    assert at.selectbox(key="case_employer").value == employer
    assert at.sidebar.radio(key="view").value == "Kasus perusahaan"
    assert "1 tiket laporan" in text(at)
    click(at, "Lihat laporan pekerja")
    assert at.sidebar.radio(key="view").value == "Laporan pekerja"
    assert set(at.dataframe[0].value.Perusahaan) == {pd.read_parquet(
        ROOT / "demo" / "B" / "observed" / "employer_registry.parquet").set_index("employer_id").at[employer, "name"]}

    at.selectbox(key="branch").set_value("C").run()
    assert not at.exception
    assert ticket not in set(at.dataframe[0].value.Tiket)
    assert reports.count("C") == 4


def test_worker_to_review_case_link_resolution_and_reopen_keeps_case_outcome_distinct(monkeypatch):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    at = app()
    cases = at.session_state["case_store"]
    reports = at.session_state["report_store"]
    tk = pd.read_parquet(ROOT / "demo" / "B" / "observed" / "tk_exchange.parquet")
    ks = pd.read_parquet(ROOT / "demo" / "B" / "observed" / "ks_membership.parquet")
    members = pd.read_parquet(ROOT / "demo" / "B" / "observed" / "ks_members.parquet")
    candidate = tk[(tk.quarter == 7) & (tk.program == "regular") & ~tk.nik.isin(ks.nik) & ~tk.nik.isin(members.nik)]
    worker = next(r for r in candidate.itertuples() if cases.get("B", r.employer_id) is None)
    before, before_c = cases.count("B"), cases.count("C")
    ranked_before = pd.read_parquet(ROOT / "demo" / "B" / "ranked.parquet")
    at.sidebar.radio(key="view").set_value("Cek kepesertaan saya").run()
    widget(at, "text_input", "NIK (16 digit)").input(worker.nik).run()
    click(at, "Kirim laporan demo")
    row = reports.reports("B").query("source == 'pekerja'").iloc[0]
    ticket = row.ticket
    assert row.employer_id == worker.employer_id
    at.sidebar.radio(key="view").set_value("Laporan pekerja").run()
    widget(at, "selectbox", "Pilih tiket").set_value(ticket).run()
    assert any(b.label == "Simpan penelaahan" for b in at.button), "The inbox needs editable inspector triage."
    initial_key = widget(at, "selectbox", "Status triase").key
    widget(at, "selectbox", "Status triase").set_value("Ditinjau")
    widget(at, "selectbox", "Pemeriksa penelaah").set_value("Pemeriksa B")
    widget(at, "text_area", "Catatan investigasi").input("Hubungi HRD untuk klarifikasi sintetis")
    click(at, "Simpan penelaahan")
    assert reports.get("B", ticket)["status"] == "Ditinjau"
    assert reports.get("B", ticket)["reviewer"] == "Pemeriksa B"
    assert widget(at, "selectbox", "Status triase").key != initial_key
    assert widget(at, "selectbox", "Status triase").value == "Ditinjau"
    assert widget(at, "text_area", "Catatan investigasi").value == ""
    click(at, "Simpan penelaahan")
    assert reports.get("B", ticket)["status"] == "Ditinjau"
    click(at, "Buat dan tautkan tindak lanjut")
    report = reports.get("B", ticket)
    case = cases.get("B", worker.employer_id)
    assert report["status"] == "Ditindaklanjuti" and report["case_id"] == case["case_id"]
    assert cases.count("B") == before + 1 and cases.count("C") == before_c
    assert ticket in cases.events(case["case_id"]).detail.iloc[-1]
    click(at, "Selesaikan laporan")
    assert reports.get("B", ticket)["status"] == "Ditindaklanjuti"
    assert "Pilih disposisi" in text(at)
    widget(at, "radio", "Disposisi laporan").set_value("diteruskan")
    click(at, "Selesaikan laporan")
    assert "Isi catatan penyelesaian" in text(at)
    widget(at, "text_area", "Catatan penyelesaian").input("Dokumen diteruskan kepada pemeriksa, belum ada kesimpulan")
    click(at, "Selesaikan laporan")
    assert reports.get("B", ticket)["status"] == "Selesai"
    assert cases.get("B", worker.employer_id) == case
    assert "Dokumen diteruskan" in text(at)
    assert "Hubungi HRD" in at.dataframe[-1].value.Keterangan.to_string(index=False)
    click(at, "Buka kembali laporan")
    assert "Isi alasan" in text(at)
    widget(at, "text_input", "Alasan membuka kembali laporan").input("Dokumen baru diterima")
    click(at, "Buka kembali laporan")
    assert reports.get("B", ticket)["status"] == "Ditindaklanjuti"
    click(at, "Buka perusahaan")
    assert at.selectbox(key="case_employer").value == worker.employer_id
    assert not at.exception
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT / "demo" / "B" / "ranked.parquet"), ranked_before)


def test_case_with_no_reports_opens_an_empty_inbox_for_that_company(monkeypatch):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    at = app()
    reports = at.session_state["report_store"]
    ranked = pd.read_parquet(ROOT / "demo" / "B" / "ranked.parquet")
    target = next(e for e in ranked.index if reports.count("B", e) == 0)
    at.sidebar.radio(key="view").set_value("Kasus perusahaan").run()
    at.selectbox(key="case_employer").set_value(target).run()
    assert "0 tiket laporan" in text(at)
    click(at, "Lihat laporan pekerja")
    assert at.dataframe[0].value.empty, "An employer deep link must not display unrelated reports."
    assert widget(at, "selectbox", "Perusahaan laporan").value == target
    assert "Belum ada laporan sesuai filter" in text(at)


def test_linking_an_existing_closed_case_never_reopens_or_replaces_its_outcome(monkeypatch):
    import hashlib

    monkeypatch.setenv("TEPAT_DEMO", "1")
    at = app()
    cases = at.session_state["case_store"]
    reports = at.session_state["report_store"]
    case = cases.cases("B").query("status == 'Selesai'").iloc[0].to_dict()
    count = cases.count("B")
    ticket = reports.submit("B", hashlib.sha256(b"closed-case-review").hexdigest(), case["employer_id"],
                            "tidak_didaftarkan").ticket
    at.sidebar.radio(key="view").set_value("Laporan pekerja").run()
    widget(at, "selectbox", "Pilih tiket").set_value(ticket).run()
    click(at, "Tautkan kasus yang ada")
    assert reports.get("B", ticket)["case_id"] == case["case_id"]
    assert cases.by_id(case["case_id"]) == case
    assert cases.count("B") == count
    assert "tidak dibuka kembali" in text(at)
    click(at, "Buka perusahaan")
    assert at.selectbox(key="case_employer").value == case["employer_id"]
    assert cases.by_id(case["case_id"]) == case
    assert not at.exception


def test_inbox_filters_do_not_change_branch_totals_or_leak_sessions(monkeypatch):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    first, second = app(), app()
    one, two = first.session_state["report_store"], second.session_state["report_store"]
    assert one.conn is not two.conn
    ticket = one.reports("B").query("status == 'Baru'").iloc[0].ticket
    one.review("B", ticket, note="Catatan hanya untuk sesi pertama")
    assert "Catatan hanya untuk sesi pertama" not in "\n".join(
        two.events("B", t).detail.to_string(index=False) for t in two.reports("B").ticket)
    first.sidebar.radio(key="view").set_value("Laporan pekerja").run()
    widget(first, "multiselect", "Status laporan").set_value(["Ditinjau"]).run()
    assert set(first.dataframe[0].value.Status) == {"Ditinjau"}
    assert first.metric[0].value == "4"
    widget(first, "multiselect", "Jenis laporan").set_value(["iuran_tidak_dibayar"]).run()
    assert len(first.dataframe[0].value) == 1
    widget(first, "multiselect", "Pemeriksa laporan").set_value(["Pemeriksa D"]).run()
    assert first.dataframe[0].value.empty and not first.exception
    assert first.metric[0].value == "4"
    first.selectbox(key="branch").set_value("C").run()
    assert len(first.dataframe[0].value) == 4 and first.metric[0].value == "4"
    assert not first.exception


def test_report_for_an_employer_missing_from_current_dataset_has_no_broken_navigation(monkeypatch):
    import hashlib

    monkeypatch.setenv("TEPAT_DEMO", "1")
    at = app()
    reports = at.session_state["report_store"]
    ticket = reports.submit("B", hashlib.sha256(b"previous-dataset").hexdigest(), "PK-MISSING",
                            "tidak_didaftarkan").ticket
    at.sidebar.radio(key="view").set_value("Laporan pekerja").run()
    widget(at, "selectbox", "Pilih tiket").set_value(ticket).run()
    assert not at.exception
    assert widget(at, "button", "Buka perusahaan").disabled
    assert "Perusahaan tidak tersedia" in text(at)
    widget(at, "text_area", "Catatan investigasi").input("Perusahaan tidak tersedia dalam cuplikan data ini")
    click(at, "Simpan penelaahan")
    assert reports.events("B", ticket).kind.iloc[-1] == "catatan"
