"""Branch-scoped worker allegations in the same SQLite database as inspector cases.

Only a reporter hash is accepted here, never a raw NIK. Hashing is a demo linkage
mechanism, not production anonymization. Reports never change model rankings.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import re
import uuid

import pandas as pd

from tepat.cases import CaseStore, INSPECTORS, UNASSIGNED, _serialized, _transactional


TYPES = {
    "tidak_didaftarkan": "Belum didaftarkan pemberi kerja",
    "iuran_tidak_dibayar": "Iuran diduga belum dibayar",
    "terdaftar_tanggungan": "Hanya tercatat sebagai tanggungan",
}
STATUSES = ["Baru", "Ditinjau", "Ditindaklanjuti", "Selesai"]
OPEN_STATUSES = STATUSES[:-1]
CLOSED = "Selesai"
DISPOSITIONS = {
    "diteruskan": "Diteruskan ke petugas terkait",
    "penelaahan_selesai": "Penelaahan administrasi selesai",
    "informasi_belum_cukup": "Informasi belum cukup",
    "di_luar_lingkup": "Di luar lingkup pemeriksaan",
}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket TEXT NOT NULL UNIQUE,
    branch TEXT NOT NULL,
    reporter_hash TEXT NOT NULL,
    employer_id TEXT NOT NULL,
    report_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_received_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    duplicate_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'Baru',
    reviewer TEXT NOT NULL,
    source TEXT NOT NULL,
    case_id INTEGER REFERENCES cases(case_id),
    resolution TEXT,
    resolution_note TEXT,
    resolved_at TEXT,
    UNIQUE (branch, reporter_hash, employer_id, report_type)
);
CREATE TABLE IF NOT EXISTS report_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id INTEGER NOT NULL REFERENCES reports(report_id),
    at TEXT NOT NULL,
    actor TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL
);
"""


class ReportError(ValueError):
    """A request that would leave a report in an invalid state."""


@dataclass(frozen=True)
class Receipt:
    ticket: str
    duplicate: bool


def hash_reporter(nik: str) -> str:
    """Demo-only stable linkage. Unsalted hashing is not production anonymization."""
    if not re.fullmatch(r"99[0-9]{14}", nik):
        raise ReportError("Gunakan NIK sintetis 16 digit yang diawali 99.")
    return hashlib.sha256(nik.encode("ascii")).hexdigest()


def _safe_text(text: str) -> str:
    if re.search(r"[0-9]{16}", text):
        raise ReportError("Jangan masukkan NIK mentah dalam catatan atau identitas petugas.")
    return text.strip()


def _timestamp(at: str | None) -> str:
    value = at if at is not None else datetime.now().isoformat(timespec="seconds")
    try:
        datetime.fromisoformat(value)
    except ValueError:
        raise ReportError("Waktu laporan harus berupa tanggal dan jam ISO.") from None
    return value


class ReportStore:
    def __init__(self, cases: CaseStore):
        self.cases = cases
        self.conn = cases.conn
        self._lock = cases._lock
        self._transaction = cases._transaction
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    @_serialized
    def get(self, branch: str, ticket: str) -> dict:
        row = self.conn.execute("SELECT * FROM reports WHERE branch = ? AND ticket = ?", (branch, ticket)).fetchone()
        if row is None:
            raise ReportError("Tiket tidak ditemukan di cabang ini.")
        return dict(row)

    @_serialized
    def events(self, branch: str, ticket: str) -> pd.DataFrame:
        report = self.get(branch, ticket)
        return pd.read_sql_query(
            "SELECT at, actor, kind, detail FROM report_events WHERE report_id = ? ORDER BY event_id",
            self.conn, params=(report["report_id"],))

    @_serialized
    def reports(self, branch: str) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM reports WHERE branch = ? ORDER BY report_id DESC",
                                 self.conn, params=(branch,))

    @_serialized
    def count(self, branch: str, employer_id: str | None = None) -> int:
        query, params = "SELECT COUNT(*) FROM reports WHERE branch = ?", [branch]
        if employer_id is not None:
            query += " AND employer_id = ?"
            params.append(employer_id)
        return int(self.conn.execute(query, params).fetchone()[0])

    def _event(self, report_id: int, actor: str, kind: str, detail: str, at: str) -> None:
        self.conn.execute("INSERT INTO report_events (report_id, at, actor, kind, detail) VALUES (?, ?, ?, ?, ?)",
                          (report_id, at, actor, kind, detail))

    @_transactional
    def submit(self, branch: str, reporter_hash: str, employer_id: str, report_type: str, *,
               source: str = "pekerja", at: str | None = None) -> Receipt:
        if branch not in {"A", "B", "C"}:
            raise ReportError("Pilih cabang data sintetis yang tersedia.")
        if not employer_id.strip() or employer_id != employer_id.strip():
            raise ReportError("Nomor pemberi kerja tidak valid.")
        _safe_text(employer_id)
        if report_type not in TYPES or source not in {"pekerja", "contoh"}:
            raise ReportError("Jenis atau sumber laporan tidak dikenal.")
        if not re.fullmatch(r"[0-9a-f]{64}", reporter_hash):
            raise ReportError("Repository hanya menerima hash pelapor, bukan NIK mentah.")
        at = _timestamp(at)
        ticket = "TPT-" + uuid.uuid4().hex[:16].upper()
        with self._transaction():
            cur = self.conn.execute(
                "INSERT INTO reports (ticket, branch, reporter_hash, employer_id, report_type, created_at, "
                "last_received_at, updated_at, reviewer, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (branch, reporter_hash, employer_id, report_type) DO NOTHING",
                (ticket, branch, reporter_hash, employer_id, report_type, at, at, at, UNASSIGNED, source))
            if cur.rowcount == 0:
                old = self.conn.execute(
                    "SELECT report_id, ticket, duplicate_count FROM reports "
                    "WHERE branch = ? AND reporter_hash = ? AND employer_id = ? AND report_type = ?",
                    (branch, reporter_hash, employer_id, report_type)).fetchone()
                self.conn.execute(
                    "UPDATE reports SET duplicate_count = duplicate_count + 1, last_received_at = ?, updated_at = ? "
                    "WHERE report_id = ?", (at, at, old["report_id"]))
                self._event(old["report_id"], "Pekerja demo", "duplikat",
                            f"Pengiriman ulang dicatat (duplikat ke-{old['duplicate_count'] + 1}). Status tidak diubah.", at)
                return Receipt(old["ticket"], True)
            assert cur.lastrowid is not None
            self._event(cur.lastrowid, "Pekerja demo", "diterima", "Laporan dugaan diterima.", at)
        return Receipt(ticket, False)

    @_transactional
    def review(self, branch: str, ticket: str, *, status: str | None = None, reviewer: str | None = None,
               note: str = "", actor: str = "Petugas", at: str | None = None) -> list[str]:
        note, actor = _safe_text(note), _safe_text(actor)
        report = self.get(branch, ticket)
        if report["status"] == CLOSED:
            raise ReportError("Laporan sudah selesai. Buka kembali sebelum mengubahnya.")
        if status is not None and status not in OPEN_STATUSES:
            raise ReportError("Pilih status terbuka. Penyelesaian memerlukan disposisi dan catatan.")
        if status == "Ditindaklanjuti" and report["case_id"] is None:
            raise ReportError("Tautkan kasus sebelum memilih status Ditindaklanjuti.")
        if reviewer is not None and reviewer not in INSPECTORS:
            raise ReportError("Pilih pemeriksa yang tersedia.")
        at = _timestamp(at)
        changes = []
        with self._transaction():
            for column, value, kind, detail in (
                ("status", status, "status", f"Status: {report['status']} menjadi {status}"),
                ("reviewer", reviewer, "pemeriksa", f"Ditugaskan kepada {reviewer}"),
            ):
                if value is not None and value != report[column]:
                    self.conn.execute(f"UPDATE reports SET {column} = ?, updated_at = ? WHERE report_id = ?",
                                      (value, at, report["report_id"]))
                    self._event(report["report_id"], actor, kind, detail, at)
                    changes.append(kind)
            if note.strip():
                self.conn.execute("UPDATE reports SET updated_at = ? WHERE report_id = ?", (at, report["report_id"]))
                self._event(report["report_id"], actor, "catatan", note.strip(), at)
                changes.append("catatan")
        return changes

    @_transactional
    def resolve(self, branch: str, ticket: str, *, disposition: str | None, note: str,
                actor: str = "Petugas", at: str | None = None) -> None:
        note, actor = _safe_text(note), _safe_text(actor)
        report = self.get(branch, ticket)
        if report["status"] == CLOSED:
            raise ReportError("Laporan sudah selesai.")
        if disposition not in DISPOSITIONS:
            raise ReportError("Pilih disposisi sebelum menyelesaikan laporan.")
        if not note.strip():
            raise ReportError("Isi catatan penyelesaian laporan.")
        at = _timestamp(at)
        with self._transaction():
            self.conn.execute(
                "UPDATE reports SET status = ?, resolution = ?, resolution_note = ?, resolved_at = ?, updated_at = ? "
                "WHERE report_id = ?", (CLOSED, disposition, note.strip(), at, at, report["report_id"]))
            self._event(report["report_id"], actor, "selesai",
                        f"Disposisi: {DISPOSITIONS[disposition]}. Catatan: {note.strip()}", at)

    @_transactional
    def link_case(self, branch: str, ticket: str, *, est_rp: float, p_violation: float, action: str,
                  actor: str = "Petugas", at: str | None = None) -> int:
        action, actor = _safe_text(action), _safe_text(actor)
        report = self.get(branch, ticket)
        if report["case_id"] is not None:
            return int(report["case_id"])
        if report["status"] == CLOSED:
            raise ReportError("Laporan sudah selesai. Buka kembali sebelum menautkan kasus.")
        at = _timestamp(at)
        with self._transaction():
            existing = self.cases.get(branch, report["employer_id"])
            if existing:
                cid = int(existing["case_id"])
            else:
                cid = self.cases.open_case(branch, report["employer_id"], est_rp=est_rp, p_violation=p_violation,
                                           action=action, actor=actor, source="laporan", at=at)
                assert cid is not None
            self.cases._event(cid, actor, "laporan ditautkan",
                              f"Tiket {ticket} ditautkan. Laporan masih dugaan; hasil kasus tidak diubah.", at)
            self.conn.execute(
                "UPDATE reports SET case_id = ?, status = 'Ditindaklanjuti', updated_at = ? WHERE report_id = ?",
                (cid, at, report["report_id"]))
            self._event(report["report_id"], actor, "kasus ditautkan",
                        f"Kasus {cid} ditautkan. Status dan hasil kasus tidak diubah.", at)
        return cid

    @_transactional
    def reopen(self, branch: str, ticket: str, *, reason: str, actor: str = "Petugas",
               at: str | None = None) -> None:
        reason, actor = _safe_text(reason), _safe_text(actor)
        report = self.get(branch, ticket)
        if report["status"] != CLOSED:
            raise ReportError("Laporan belum selesai.")
        if not reason.strip():
            raise ReportError("Isi alasan membuka kembali laporan.")
        at = _timestamp(at)
        status = "Ditindaklanjuti" if report["case_id"] is not None else "Ditinjau"
        with self._transaction():
            self.conn.execute(
                "UPDATE reports SET status = ?, resolution = NULL, resolution_note = NULL, resolved_at = NULL, "
                "updated_at = ? WHERE report_id = ?", (status, at, report["report_id"]))
            self._event(report["report_id"], actor, "dibuka kembali", reason.strip(), at)


def seed_examples(store: ReportStore, branch: str, ranked: pd.DataFrame, today: date | None = None) -> int:
    """Four private examples. No answer-key access, and never seed a persistent database."""
    if store.cases.mode != "sandbox" or store.count(branch) or len(ranked) < 4:
        return 0
    today = today or date.today()

    def stamp(days_ago: int, hour: int = 9) -> str:
        return datetime.combine(today - timedelta(days=days_ago), datetime.min.time()).replace(
            hour=hour).isoformat(timespec="seconds")

    employers = list(ranked.sort_values("rank").index[:4])
    existing = store.cases.cases(branch)
    if not existing.empty:
        # Reuse an example case rather than creating another board row on every app visit.
        candidates = existing[~existing.employer_id.isin([employers[0], employers[1], employers[3]])]
        if not candidates.empty:
            employers[2] = candidates.iloc[0].employer_id
    for i, eid in enumerate(employers):
        reporter = hashlib.sha256(f"contoh:{branch}:{i}".encode()).hexdigest()
        kind = list(TYPES)[i % len(TYPES)]
        ticket = store.submit(branch, reporter, str(eid), kind, source="contoh", at=stamp(8 - i)).ticket
        if i == 1:
            store.review(branch, ticket, status="Ditinjau", reviewer="Pemeriksa A",
                         note="Contoh: minta klarifikasi administrasi. Belum ada kesimpulan pemeriksaan.",
                         actor="Contoh demo", at=stamp(3))
            store.submit(branch, reporter, str(eid), kind, source="contoh", at=stamp(1))
        elif i == 2:
            r = ranked.loc[eid]
            store.link_case(branch, ticket, est_rp=r.est_rp, p_violation=r.p_violation, action=r.action,
                            actor="Contoh demo", at=stamp(2))
        elif i == 3:
            store.resolve(branch, ticket, disposition="informasi_belum_cukup",
                          note="Contoh: dokumen tambahan belum tersedia. Ini bukan penetapan pelanggaran.",
                          actor="Contoh demo", at=stamp(1))
    return 4
