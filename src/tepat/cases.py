"""Case follow-up and outcome tracking for inspectors (synthetic demo).

A case starts only when an inspector asks for it (one employer, or every employer inside the hour
budget). It then carries a status, an assigned inspector, a follow-up date, notes and a dated history.
Closing a case needs an outcome. Recorded outcomes can be exported as labels for a later retraining
run; nothing here retrains the model or sends anything to an employer.

Storage is SQLite. `CaseStore.persistent(path)` keeps a file (local runs). `CaseStore.sandbox()` keeps
an in-memory database, so each visitor of the public demo gets a private copy.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from functools import wraps
from pathlib import Path
from threading import RLock

import pandas as pd

STATUSES = ["Baru", "Ditugaskan", "Surat dikirim", "Pemeriksaan dijadwalkan", "Menunggu koreksi", "Selesai"]
OPEN_STATUSES = STATUSES[:-1]
CLOSED = "Selesai"
OUTCOMES = {
    "pelanggaran_terbukti": "Pelanggaran terbukti",
    "koreksi_mandiri": "Dikoreksi mandiri",
    "cicilan": "Masuk program cicilan",
    "tidak_terbukti": "Tidak ditemukan pelanggaran",
}
VIOLATION_OUTCOMES = {"pelanggaran_terbukti", "koreksi_mandiri", "cicilan"}
INSPECTORS = ["Belum ditugaskan", "Pemeriksa A", "Pemeriksa B", "Pemeriksa C", "Pemeriksa D"]
UNASSIGNED = INSPECTORS[0]
DEFAULT_FOLLOW_UP_DAYS = 14
# Approving a draft (feature #2) suggests the next status. Nothing is sent.
DRAFT_NEXT_STATUS = {
    "brief": "Pemeriksaan dijadwalkan",
    "surat_kepatuhan": "Surat dikirim",
    "surat_cicilan": "Surat dikirim",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    branch       TEXT NOT NULL,
    employer_id  TEXT NOT NULL,
    status       TEXT NOT NULL,
    owner        TEXT NOT NULL,
    due_date     TEXT,
    opened_at    TEXT NOT NULL,
    closed_at    TEXT,
    outcome      TEXT,
    recovered_rp REAL,
    est_rp       REAL,
    p_violation  REAL,
    action       TEXT,
    source       TEXT NOT NULL,
    UNIQUE (branch, employer_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id  INTEGER NOT NULL REFERENCES cases(case_id),
    at       TEXT NOT NULL,
    actor    TEXT NOT NULL,
    kind     TEXT NOT NULL,
    detail   TEXT NOT NULL
);
"""


class CaseError(ValueError):
    """A request that would leave a case in an invalid state."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _iso(d: date | str | None) -> str | None:
    if d is None or d == "":
        return None
    return d if isinstance(d, str) else d.isoformat()


def _serialized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


def _transactional(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._transaction():
            return method(self, *args, **kwargs)
    return call


class CaseStore:
    def __init__(self, conn: sqlite3.Connection, mode: str, path: Path | None = None):
        self.conn = conn
        self.mode = mode
        self.path = path
        self._lock = RLock()
        self._transaction_depth = 0
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    @contextmanager
    def _transaction(self):
        """Serialize validation and writes, including nested report/case work.

        The lock protects the shared connection. BEGIN IMMEDIATE orders writers
        on other connections before they read mutable state. Only the outermost
        operation commits or rolls back.
        """
        with self._lock:
            outermost = self._transaction_depth == 0
            if outermost:
                self.conn.execute("BEGIN IMMEDIATE")
            self._transaction_depth += 1
            try:
                yield
                if outermost:
                    self.conn.commit()
            except BaseException:
                if outermost:
                    self.conn.rollback()
                raise
            finally:
                self._transaction_depth -= 1

    @classmethod
    def persistent(cls, path: str | Path) -> "CaseStore":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite3.connect(str(path), check_same_thread=False), "persistent", path)

    @classmethod
    def sandbox(cls) -> "CaseStore":
        return cls(sqlite3.connect(":memory:", check_same_thread=False), "sandbox")

    # ------------------------------------------------------------------ reads
    @_serialized
    def get(self, branch: str, employer_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM cases WHERE branch = ? AND employer_id = ?", (branch, employer_id)).fetchone()
        return dict(row) if row else None

    @_serialized
    def by_id(self, case_id: int) -> dict:
        row = self.conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            raise CaseError(f"Kasus {case_id} tidak ditemukan.")
        return dict(row)

    @_serialized
    def cases(self, branch: str) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM cases WHERE branch = ? ORDER BY case_id", self.conn, params=(branch,))

    @_serialized
    def events(self, case_id: int) -> pd.DataFrame:
        return pd.read_sql_query("SELECT at, actor, kind, detail FROM events WHERE case_id = ? ORDER BY event_id",
                                 self.conn, params=(case_id,))

    @_serialized
    def count(self, branch: str) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM cases WHERE branch = ?", (branch,)).fetchone()[0])

    # ------------------------------------------------------------------ writes
    def _event(self, case_id: int, actor: str, kind: str, detail: str, at: str | None = None) -> None:
        self.conn.execute("INSERT INTO events (case_id, at, actor, kind, detail) VALUES (?, ?, ?, ?, ?)",
                          (case_id, at or _now(), actor, kind, detail))

    @_transactional
    def open_case(self, branch: str, employer_id: str, *, est_rp: float, p_violation: float, action: str,
                  actor: str = "Petugas", source: str = "petugas", due: date | None = None,
                  at: str | None = None) -> int | None:
        """Start follow-up, or return None when the employer already has a case.
        Nested calls participate in the enclosing report-link transaction."""
        if self.get(branch, employer_id) is not None:
            return None
        due = due or (date.today() + timedelta(days=DEFAULT_FOLLOW_UP_DAYS))
        cur = self.conn.execute(
            "INSERT INTO cases (branch, employer_id, status, owner, due_date, opened_at, est_rp, p_violation, action, source) "
            "VALUES (?, ?, 'Baru', ?, ?, ?, ?, ?, ?, ?)",
            (branch, employer_id, UNASSIGNED, _iso(due), at or _now(), float(est_rp), float(p_violation), action, source))
        case_id = int(cur.lastrowid)
        self._event(case_id, actor, "dibuka", f"Tindak lanjut dimulai. Saran sistem: {action}", at)
        return case_id

    @_transactional
    def bulk_open(self, branch: str, rows: pd.DataFrame, *, actor: str = "Petugas") -> tuple[int, int]:
        """Open a case for every row (index = employer_id; columns est_rp, p_violation, action).
        Employers that already have a case, open or closed, are skipped. Returns (created, skipped)."""
        created = skipped = 0
        for eid, r in rows.iterrows():
            cid = self.open_case(branch, str(eid), est_rp=r.est_rp, p_violation=r.p_violation, action=r.action,
                                 actor=actor, source="massal")
            created += cid is not None
            skipped += cid is None
        return created, skipped

    @_transactional
    def update(self, case_id: int, *, status: str | None = None, owner: str | None = None,
               due: date | str | None = None, note: str | None = None, actor: str = "Petugas",
               at: str | None = None) -> list[str]:
        """Change status, inspector or follow-up date and/or add a note. Returns the changes recorded."""
        case = self.by_id(case_id)
        if case["status"] == CLOSED:
            raise CaseError("Kasus sudah selesai. Buka kembali sebelum mengubahnya.")
        changes = []
        if status in (None, "Baru") and case["status"] == "Baru" and owner not in (None, UNASSIGNED):
            status = "Ditugaskan"  # assigning an inspector moves a new case forward
        if status is not None and status != case["status"]:
            if status == CLOSED:
                raise CaseError("Kasus hanya dapat diselesaikan dengan mencatat hasil.")
            if status not in OPEN_STATUSES:
                raise CaseError(f"Status tidak dikenal: {status}")
            self.conn.execute("UPDATE cases SET status = ? WHERE case_id = ?", (status, case_id))
            self._event(case_id, actor, "status", f"Status: {case['status']} menjadi {status}", at)
            changes.append("status")
        if owner is not None and owner != case["owner"]:
            if owner not in INSPECTORS:
                raise CaseError(f"Pemeriksa tidak dikenal: {owner}")
            self.conn.execute("UPDATE cases SET owner = ? WHERE case_id = ?", (owner, case_id))
            self._event(case_id, actor, "pemeriksa", f"Ditugaskan kepada {owner}", at)
            changes.append("pemeriksa")
        due_s = _iso(due)
        if due_s is not None and due_s != case["due_date"]:
            self.conn.execute("UPDATE cases SET due_date = ? WHERE case_id = ?", (due_s, case_id))
            self._event(case_id, actor, "jadwal", f"Tanggal tindak lanjut: {due_s}", at)
            changes.append("jadwal")
        if note and note.strip():
            self._event(case_id, actor, "catatan", note.strip(), at)
            changes.append("catatan")
        return changes

    @_transactional
    def close(self, case_id: int, *, outcome: str | None, recovered_rp: float | None = None, note: str = "",
              actor: str = "Petugas", at: str | None = None) -> None:
        case = self.by_id(case_id)
        if case["status"] == CLOSED:
            raise CaseError("Kasus sudah selesai.")
        if outcome not in OUTCOMES:
            raise CaseError("Pilih hasil pemeriksaan sebelum menutup kasus.")
        if outcome == "tidak_terbukti":
            recovered_rp = 0.0
        if recovered_rp is None or recovered_rp < 0:
            raise CaseError("Isi Rp yang dipulihkan (0 bila belum ada).")
        at = at or _now()
        self.conn.execute("UPDATE cases SET status = ?, outcome = ?, recovered_rp = ?, closed_at = ? WHERE case_id = ?",
                          (CLOSED, outcome, float(recovered_rp), at, case_id))
        detail = f"Selesai: {OUTCOMES[outcome]}. Rp dipulihkan {float(recovered_rp):,.0f}".replace(",", ".")
        self._event(case_id, actor, "selesai", detail + (f". {note.strip()}" if note.strip() else ""), at)

    @_transactional
    def reopen(self, case_id: int, *, reason: str, actor: str = "Petugas") -> None:
        case = self.by_id(case_id)
        if case["status"] != CLOSED:
            raise CaseError("Kasus belum selesai.")
        if not reason.strip():
            raise CaseError("Tuliskan alasan membuka kembali.")
        self.conn.execute("UPDATE cases SET status = 'Menunggu koreksi', outcome = NULL, recovered_rp = NULL, "
                          "closed_at = NULL WHERE case_id = ?", (case_id,))
        self._event(case_id, actor, "dibuka kembali", reason.strip())

    @_transactional
    def record_draft_approval(self, branch: str, employer_id: str, doc_type: str, label: str,
                              actor: str = "Petugas") -> str | None:
        """Hook for feature #2: log an approved draft in the case history.
        Returns the suggested next status, or None when the employer has no case yet."""
        case = self.get(branch, employer_id)
        if case is None:
            return None
        self._event(case["case_id"], actor, "draf disetujui", f"{label} disetujui petugas. Tidak dikirim oleh sistem.")
        nxt = DRAFT_NEXT_STATUS.get(doc_type)
        return nxt if nxt and case["status"] != CLOSED and nxt != case["status"] else None

    # ------------------------------------------------------------------ outcomes
    @_serialized
    def summary(self, branch: str) -> dict:
        C = self.cases(branch)
        done = C[C.status == CLOSED]
        viol = done[done.outcome.isin(VIOLATION_OUTCOMES)]
        return {
            "total": len(C), "open": int((C.status != CLOSED).sum()), "closed": len(done),
            "confirmed": len(viol), "false_alarms": int((done.outcome == "tidak_terbukti").sum()),
            "hit_rate": (len(viol) / len(done)) if len(done) else None,
            "recovered_rp": float(done.recovered_rp.fillna(0).sum()),
            "estimated_rp_closed": float(done.est_rp.fillna(0).sum()),
            "overdue": int(((C.status != CLOSED) & (C.due_date < date.today().isoformat())).sum()),
        }

    @_serialized
    def export_labels(self, branch: str) -> pd.DataFrame:
        """Closed cases as future training labels (1 = violation confirmed). Not used for training here."""
        C = self.cases(branch)
        C = C[C.status == CLOSED].copy()
        return pd.DataFrame({
            "branch": C.branch, "employer_id": C.employer_id, "label_violation": C.outcome.isin(VIOLATION_OUTCOMES).astype(int),
            "outcome": C.outcome, "recovered_rp": C.recovered_rp, "est_rp_tepat": C.est_rp,
            "p_violation_tepat": C.p_violation, "opened_at": C.opened_at, "closed_at": C.closed_at,
            "source": C.source,
        }).reset_index(drop=True)


def seed_examples(store: CaseStore, branch: str, ranked: pd.DataFrame, truth: pd.DataFrame,
                  today: date | None = None) -> int:
    """Example cases at every stage so the demo starts with a realistic board.

    Outcomes of the closed examples come from the synthetic answer key (truth), which a real inspector
    would only learn by inspecting; the app labels these rows as examples. Deterministic: picks by rank.
    Returns how many cases were created (0 when the branch already has cases)."""
    if store.count(branch):
        return 0
    today = today or date.today()
    if "employer_id" in truth.columns:
        truth = truth.set_index("employer_id")
    T = truth.reindex(ranked.index)
    insp = ranked[ranked.queue == "inspeksi"].sort_values("rank")
    viol = [e for e in insp.index if bool(T.at[e, "is_violator"])]
    honest = [e for e in insp.index[:150] if not bool(T.at[e, "is_violator"])]
    cic_all = list(ranked[ranked.queue == "cicilan"].sort_values("rank").index)
    cic = [e for e in cic_all if bool(T.at[e, "is_violator"])] or cic_all

    def stamp(days_ago: int, hour: int = 9) -> str:
        return datetime.combine(today - timedelta(days=days_ago), datetime.min.time()).replace(hour=hour).isoformat(timespec="seconds")

    def opened(eid: str, days_ago: int) -> int:
        r = ranked.loc[eid]
        return store.open_case(branch, eid, est_rp=r.est_rp, p_violation=r.p_violation, action=r.action,
                               actor="Contoh demo", source="contoh", due=today + timedelta(days=14 - days_ago),
                               at=stamp(days_ago))

    plan = []  # (employer, days ago opened, steps, outcome)
    stages = [
        ("Baru", []),
        ("Ditugaskan", [("owner", "Pemeriksa A")]),
        ("Surat dikirim", [("owner", "Pemeriksa B"), ("status", "Surat dikirim")]),
        ("Pemeriksaan dijadwalkan", [("owner", "Pemeriksa A"), ("status", "Pemeriksaan dijadwalkan")]),
        ("Menunggu koreksi", [("owner", "Pemeriksa C"), ("status", "Surat dikirim"), ("status", "Menunggu koreksi")]),
    ]
    pool = iter(viol[2:])
    for i, (_, steps) in enumerate(stages):
        eid = next(pool, None)
        if eid is not None:
            plan.append((eid, 2 + 3 * i, steps, None))
    if len(viol) >= 2:
        plan.append((viol[0], 30, [("owner", "Pemeriksa B"), ("status", "Pemeriksaan dijadwalkan")], "pelanggaran_terbukti"))
        plan.append((viol[1], 24, [("owner", "Pemeriksa A"), ("status", "Surat dikirim")], "koreksi_mandiri"))
    if honest:
        plan.append((honest[0], 21, [("owner", "Pemeriksa C"), ("status", "Pemeriksaan dijadwalkan")], "tidak_terbukti"))
    if cic:
        plan.append((cic[0], 18, [("owner", "Pemeriksa D"), ("status", "Surat dikirim")], "cicilan"))

    made = 0
    for eid, ago, steps, outcome in plan:
        cid = opened(eid, ago)
        if cid is None:
            continue
        made += 1
        for k, (field, value) in enumerate(steps):
            store.update(cid, actor="Contoh demo", at=stamp(max(ago - 1 - k, 0), 10), **{field: value})
        if outcome:
            rec = 0.0 if outcome == "tidak_terbukti" else float(T.at[eid, "recoverable_rp_true"])
            store.close(cid, outcome=outcome, recovered_rp=rec, actor="Contoh demo", at=stamp(max(ago - 10, 1), 15),
                        note="Contoh: hasil diambil dari kunci jawaban sintetis.")
    return made
