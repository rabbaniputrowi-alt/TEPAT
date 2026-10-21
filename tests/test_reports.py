"""Worker allegations stay branch-scoped and separate from inspection outcomes."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util

import pytest

from tepat.cases import CaseStore


def reports_module():
    assert importlib.util.find_spec("tepat.reports") is not None, "Worker reports need a durable repository."
    return importlib.import_module("tepat.reports")


def submit(repo, branch="B", reporter="synthetic-reporter", employer="PK1", kind="tidak_didaftarkan", **kwargs):
    return repo.submit(branch, hashlib.sha256(reporter.encode()).hexdigest(), employer, kind, **kwargs)


def test_report_survives_connection_recreation(tmp_path):
    R = reports_module()
    path = tmp_path / "cases.sqlite"
    cases = CaseStore.persistent(path)
    reports = R.ReportStore(cases)
    receipt = submit(reports, at="2026-10-21T09:00:00")
    assert not receipt.duplicate
    cases.conn.close()

    reopened_cases = CaseStore.persistent(path)
    reopened = R.ReportStore(reopened_cases)
    report = reopened.get("B", receipt.ticket)
    assert report["ticket"].startswith("TPT-")
    assert report["employer_id"] == "PK1"
    assert report["status"] == "Baru"
    assert report["duplicate_count"] == 0
    assert report["created_at"] == report["last_received_at"] == "2026-10-21T09:00:00"
    assert reopened.events("B", receipt.ticket).kind.tolist() == ["diterima"]
    assert reopened_cases.count("B") == 0
    reopened_cases.conn.close()
    assert {p.name for p in tmp_path.iterdir()} == {"cases.sqlite"}


def test_duplicate_submission_reuses_ticket_preserves_evidence_and_branch_scope():
    R = reports_module()
    cases = CaseStore.sandbox()
    reports = R.ReportStore(cases)
    first = submit(reports, at="2026-10-20T09:00:00")
    second = submit(reports, at="2026-10-21T10:00:00")
    assert second.duplicate and second.ticket == first.ticket
    report = reports.get("B", first.ticket)
    assert report["duplicate_count"] == 1
    assert report["created_at"] == "2026-10-20T09:00:00"
    assert report["last_received_at"] == "2026-10-21T10:00:00"
    assert reports.events("B", first.ticket).kind.tolist() == ["diterima", "duplikat"]
    assert reports.events("B", first.ticket)["at"].tolist() == ["2026-10-20T09:00:00", "2026-10-21T10:00:00"]
    distinct = [submit(reports, branch="C"), submit(reports, reporter="another-reporter"),
                submit(reports, kind="terdaftar_tanggungan"), submit(reports, employer="PK2")]
    assert len({first.ticket, *(r.ticket for r in distinct)}) == 5
    assert reports.count("B") == 4 and reports.count("C") == 1
    assert reports.count("B", "PK1") == 3
    assert len(reports.reports("B")) == 4
    with pytest.raises(R.ReportError):
        reports.get("C", first.ticket)
    with pytest.raises(R.ReportError):
        reports.events("C", first.ticket)
    assert cases.count("B") == cases.count("C") == 0


def test_review_records_reviewer_status_and_dated_notes_without_partial_invalid_writes():
    R = reports_module()
    reports = R.ReportStore(CaseStore.sandbox())
    ticket = submit(reports).ticket
    assert hasattr(reports, "review"), "Inspectors need a recorded review workflow."
    changes = reports.review("B", ticket, status="Ditinjau", reviewer="Pemeriksa B", note="Minta daftar gaji sintetis",
                             at="2026-10-21T11:00:00")
    assert changes == ["status", "pemeriksa", "catatan"]
    report = reports.get("B", ticket)
    assert (report["status"], report["reviewer"]) == ("Ditinjau", "Pemeriksa B")
    events = reports.events("B", ticket)
    assert events.kind.tolist() == ["diterima", "status", "pemeriksa", "catatan"]
    assert events.iloc[-1].detail == "Minta daftar gaji sintetis"
    assert events.iloc[-1]["at"] == "2026-10-21T11:00:00"
    for kwargs in ({"status": "Selesai"}, {"status": "Tidak valid"},
                   {"status": "Baru", "reviewer": "Tidak valid"}, {"status": "Ditindaklanjuti"}):
        with pytest.raises(R.ReportError):
            reports.review("B", ticket, **kwargs)
        assert reports.get("B", ticket) == report
        assert len(reports.events("B", ticket)) == len(events)
    with pytest.raises(R.ReportError):
        reports.review("C", ticket, note="Cabang lain")
    assert reports.review("B", ticket, status="Ditinjau", reviewer="Pemeriksa B", note="  ") == []


def test_resolution_requires_valid_disposition_and_note_and_preserves_closed_report_on_duplicate():
    R = reports_module()
    reports = R.ReportStore(CaseStore.sandbox())
    ticket = submit(reports).ticket
    assert hasattr(reports, "resolve"), "Resolution needs an explicit disposition and note."
    before = reports.get("B", ticket)
    for disposition, note in ((None, "Catatan"), ("invalid", "Catatan"), ("informasi_belum_cukup", "  ")):
        with pytest.raises(R.ReportError):
            reports.resolve("B", ticket, disposition=disposition, note=note)
        assert reports.get("B", ticket) == before
        assert len(reports.events("B", ticket)) == 1
    with pytest.raises(R.ReportError):
        reports.resolve("C", ticket, disposition="informasi_belum_cukup", note="Cabang lain")
    reports.resolve("B", ticket, disposition="informasi_belum_cukup", note="Perlu dokumen tambahan dari pekerja",
                    at="2026-10-21T12:00:00")
    closed = reports.get("B", ticket)
    assert closed["status"] == "Selesai" and closed["resolution"] == "informasi_belum_cukup"
    assert closed["resolution_note"] == "Perlu dokumen tambahan dari pekerja"
    assert closed["resolved_at"] == "2026-10-21T12:00:00"
    assert "Perlu dokumen tambahan" in reports.events("B", ticket).detail.iloc[-1]
    assert reports.cases.count("B") == 0
    with pytest.raises(R.ReportError):
        reports.review("B", ticket, note="Tidak boleh mengubah laporan selesai")
    with pytest.raises(R.ReportError):
        reports.resolve("B", ticket, disposition="informasi_belum_cukup", note="Ulangi")
    assert submit(reports).duplicate
    after = reports.get("B", ticket)
    for field in ("status", "resolution", "resolution_note", "resolved_at", "case_id"):
        assert after[field] == closed[field]


@pytest.mark.parametrize("existing", [None, "open", "closed"])
def test_explicit_case_link_is_idempotent_and_never_changes_existing_outcome(existing):
    R = reports_module()
    cases = CaseStore.sandbox()
    reports = R.ReportStore(cases)
    ticket = submit(reports).ticket
    if existing:
        cid = cases.open_case("B", "PK1", est_rp=123, p_violation=0.2, action="Tinjau")
        cases.update(cid, owner="Pemeriksa A", note="Bukti lama")
        if existing == "closed":
            cases.close(cid, outcome="tidak_terbukti", note="Hasil pemeriksaan lama")
        before = cases.by_id(cid)
        old_events = len(cases.events(cid))
    else:
        before, old_events = None, 0
    assert hasattr(reports, "link_case"), "Report-to-case follow-up must be explicit."
    with pytest.raises(R.ReportError):
        reports.link_case("C", ticket, est_rp=999, p_violation=0.8, action="Periksa")
    linked = reports.link_case("B", ticket, est_rp=999, p_violation=0.8, action="Periksa",
                               at="2026-10-21T13:00:00")
    assert cases.count("B") == 1 and cases.count("C") == 0
    case = cases.by_id(linked)
    if before:
        assert case == before
    else:
        assert case["status"] == "Baru" and case["source"] == "laporan"
    report = reports.get("B", ticket)
    assert report["case_id"] == linked and report["status"] == "Ditindaklanjuti"
    assert ticket in cases.events(linked).detail.iloc[-1]
    assert reports.events("B", ticket).kind.iloc[-1] == "kasus ditautkan"
    case_event_count, report_event_count = len(cases.events(linked)), len(reports.events("B", ticket))
    assert case_event_count == old_events + (1 if before else 2)
    assert reports.link_case("B", ticket, est_rp=0, p_violation=0, action="Tidak mengubah") == linked
    assert len(cases.events(linked)) == case_event_count
    assert len(reports.events("B", ticket)) == report_event_count
    assert submit(reports).duplicate
    assert cases.count("B") == 1
    reports.resolve("B", ticket, disposition="diteruskan", note="Diserahkan kepada pemeriksa")
    assert cases.by_id(linked) == case
    assert not cases.export_labels("B").empty if existing == "closed" else cases.export_labels("B").empty


def test_case_creation_and_report_link_roll_back_as_one_transaction():
    import sqlite3

    R = reports_module()
    cases = CaseStore.sandbox()
    reports = R.ReportStore(cases)
    ticket = submit(reports).ticket
    cases.conn.execute("CREATE TRIGGER reject_link BEFORE UPDATE OF case_id ON reports "
                       "BEGIN SELECT RAISE(ABORT, 'synthetic injected failure'); END")
    cases.conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        reports.link_case("B", ticket, est_rp=1, p_violation=0.2, action="Tinjau")
    assert cases.count("B") == 0, "A failed link must not leave a newly committed case."
    assert cases.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert reports.get("B", ticket)["case_id"] is None
    assert reports.get("B", ticket)["status"] == "Baru"
    assert len(reports.events("B", ticket)) == 1


@pytest.mark.parametrize("linked", [False, True])
def test_reopen_requires_reason_preserves_resolution_history_and_case_state(linked):
    R = reports_module()
    cases = CaseStore.sandbox()
    reports = R.ReportStore(cases)
    ticket = submit(reports).ticket
    if linked:
        reports.link_case("B", ticket, est_rp=1, p_violation=0.2, action="Tinjau")
    assert hasattr(reports, "reopen"), "A resolved allegation needs an explicit reason to reopen."
    with pytest.raises(R.ReportError):
        reports.reopen("B", ticket, reason="Belum selesai")
    reports.resolve("B", ticket, disposition="informasi_belum_cukup", note="Slip gaji belum lengkap")
    old_case = cases.get("B", "PK1")
    with pytest.raises(R.ReportError):
        reports.reopen("B", ticket, reason="  ")
    with pytest.raises(R.ReportError):
        reports.reopen("C", ticket, reason="Cabang lain")
    reports.reopen("B", ticket, reason="Dokumen tambahan diterima", at="2026-10-22T09:00:00")
    report = reports.get("B", ticket)
    assert report["status"] == ("Ditindaklanjuti" if linked else "Ditinjau")
    assert report["resolution"] is report["resolution_note"] is report["resolved_at"] is None
    events = reports.events("B", ticket)
    assert "Slip gaji belum lengkap" in events.detail.iloc[-2]
    assert events.kind.iloc[-1] == "dibuka kembali" and events.detail.iloc[-1] == "Dokumen tambahan diterima"
    assert events.iloc[-1]["at"] == "2026-10-22T09:00:00"
    assert cases.get("B", "PK1") == old_case


def test_repository_accepts_only_reporter_hash_and_refuses_nik_in_notes(tmp_path):
    R = reports_module()
    assert hasattr(R, "hash_reporter"), "Hash synthetic NIKs before they reach the repository."
    raw = "9900000000000001"
    reporter_hash = R.hash_reporter(raw)
    assert reporter_hash == hashlib.sha256(raw.encode("ascii")).hexdigest()
    for bad in ("3200000000000001", "99", "９９０００００００００００００１"):
        with pytest.raises(R.ReportError):
            R.hash_reporter(bad)
    path = tmp_path / "cases.sqlite"
    cases = CaseStore.persistent(path)
    reports = R.ReportStore(cases)
    with pytest.raises(R.ReportError):
        reports.submit("B", raw, "PK1", "tidak_didaftarkan")
    ticket = reports.submit("B", reporter_hash, "PK1", "tidak_didaftarkan").ticket
    for kwargs in ({"note": raw}, {"note": "NIK " + raw}, {"actor": raw, "note": "Catatan"}):
        with pytest.raises(R.ReportError):
            reports.review("B", ticket, **kwargs)
    with pytest.raises(R.ReportError):
        reports.resolve("B", ticket, disposition="informasi_belum_cukup", note=raw)
    reports.resolve("B", ticket, disposition="informasi_belum_cukup", note="Data tambahan diperlukan")
    with pytest.raises(R.ReportError):
        reports.reopen("B", ticket, reason=raw)
    assert raw not in ticket and reporter_hash not in ticket
    history = reports.events("B", ticket).to_csv(index=False)
    assert raw not in history and reporter_hash not in history
    cases.conn.close()
    assert raw.encode() not in path.read_bytes()
    assert {p.name for p in tmp_path.iterdir()} == {"cases.sqlite"}


@pytest.mark.parametrize("overrides", [
    {"branch": ""}, {"branch": "X"}, {"employer_id": " "}, {"employer_id": "9900000000000001"},
    {"report_type": "unknown"}, {"source": "unknown"}, {"at": "not-a-time"},
])
def test_invalid_report_fields_do_not_write_partial_records(overrides):
    R = reports_module()
    reports = R.ReportStore(CaseStore.sandbox())
    values = dict(branch="B", reporter_hash=hashlib.sha256(b"demo").hexdigest(), employer_id="PK1",
                  report_type="tidak_didaftarkan")
    values.update(overrides)
    with pytest.raises(R.ReportError):
        reports.submit(**values)
    assert reports.conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0] == 0
    assert reports.conn.execute("SELECT COUNT(*) FROM report_events").fetchone()[0] == 0


def test_examples_only_seed_private_sandboxes_and_never_persistent_storage(tmp_path):
    from datetime import date
    from pathlib import Path

    import pandas as pd

    R = reports_module()
    assert hasattr(R, "seed_examples"), "Each visitor needs labelled triage examples, not shared reports."
    root = Path(__file__).resolve().parents[1]
    ranked = pd.read_parquet(root / "demo" / "B" / "ranked.parquet")
    one, two = R.ReportStore(CaseStore.sandbox()), R.ReportStore(CaseStore.sandbox())
    for reports in (one, two):
        assert R.seed_examples(reports, "B", ranked, today=date(2026, 10, 21)) == 4
        assert R.seed_examples(reports, "B", ranked) == 0
        rows = reports.reports("B")
        assert set(rows.source) == {"contoh"}
        assert set(rows.status) == set(R.STATUSES)
        assert rows.duplicate_count.sum() == 1
        assert (rows.created_at <= "2026-10-21T23:59:59").all()
        histories = "\n".join(reports.events("B", t).to_csv(index=False) for t in rows.ticket)
        assert "kunci jawaban" not in histories and "is_violator" not in histories
    ticket = one.reports("B").query("status == 'Baru'").iloc[0].ticket
    one.review("B", ticket, note="Catatan pribadi sesi satu")
    assert "Catatan pribadi sesi satu" not in "\n".join(
        two.events("B", t).to_csv(index=False) for t in two.reports("B").ticket)
    with pytest.raises(R.ReportError):
        two.get("B", ticket)
    assert one.count("C") == two.count("C") == 0
    local = R.ReportStore(CaseStore.persistent(tmp_path / "cases.sqlite"))
    assert R.seed_examples(local, "B", ranked) == 0
    assert local.count("B") == local.cases.count("B") == 0
    local.conn.close()


def test_case_link_rejects_raw_identifiers_and_invalid_event_times_before_writing():
    R = reports_module()
    reports = R.ReportStore(CaseStore.sandbox())
    ticket = submit(reports).ticket
    for kwargs in ({"actor": "9900000000000001"}, {"action": "9900000000000001"}, {"at": "not-a-time"}):
        values = dict(est_rp=1, p_violation=0.2, action="Tinjau")
        values.update(kwargs)
        with pytest.raises(R.ReportError):
            reports.link_case("B", ticket, **values)
        assert reports.cases.count("B") == 0
        assert reports.get("B", ticket)["case_id"] is None
    with pytest.raises(R.ReportError):
        reports.review("B", ticket, status="Ditinjau", at="not-a-time")
    assert reports.get("B", ticket)["status"] == "Baru"
    with pytest.raises(R.ReportError):
        reports.resolve("B", ticket, disposition="informasi_belum_cukup", note="Perlu dokumen", at="not-a-time")
    assert reports.get("B", ticket)["status"] == "Baru"
    assert len(reports.events("B", ticket)) == 1
