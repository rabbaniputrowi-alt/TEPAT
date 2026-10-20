"""Case follow-up store: lifecycle rules, bulk start, outcomes, labels and the example seed."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tepat import cases as K

ROOT = Path(__file__).resolve().parents[1]
DEMO_B = ROOT / "demo" / "B"


def _ranked():
    return pd.read_parquet(DEMO_B / "ranked.parquet")


def _truth():
    return pd.read_parquet(DEMO_B / "truth" / "employers_truth.parquet").set_index("employer_id")


def _open(store, eid="PK1", branch="B", est=10e6, p=0.9):
    return store.open_case(branch, eid, est_rp=est, p_violation=p, action="Pemeriksaan lapangan")


def test_case_lifecycle_records_every_change():
    s = K.CaseStore.sandbox()
    cid = _open(s)
    assert s.get("B", "PK1")["status"] == "Baru"
    changes = s.update(cid, owner="Pemeriksa A", status="Ditugaskan", due=date(2026, 11, 5), note="Hubungi HRD")
    assert changes == ["status", "pemeriksa", "jadwal", "catatan"]
    s.close(cid, outcome="pelanggaran_terbukti", recovered_rp=12_500_000)
    case = s.by_id(cid)
    assert case["status"] == "Selesai" and case["recovered_rp"] == 12_500_000
    kinds = s.events(cid).kind.tolist()
    assert kinds == ["dibuka", "status", "pemeriksa", "jadwal", "catatan", "selesai"]


def test_closing_needs_an_outcome_and_status_cannot_jump_to_closed():
    s = K.CaseStore.sandbox()
    cid = _open(s)
    with pytest.raises(K.CaseError):
        s.close(cid, outcome=None, recovered_rp=0)
    with pytest.raises(K.CaseError):
        s.update(cid, status="Selesai")
    with pytest.raises(K.CaseError):
        s.close(cid, outcome="koreksi_mandiri", recovered_rp=None)
    s.close(cid, outcome="tidak_terbukti", recovered_rp=999)
    assert s.by_id(cid)["recovered_rp"] == 0  # a false alarm recovers nothing
    with pytest.raises(K.CaseError):
        s.update(cid, note="terlambat")
    s.reopen(cid, reason="Dokumen baru masuk")
    assert s.by_id(cid)["status"] == "Menunggu koreksi" and s.by_id(cid)["outcome"] is None


def test_one_case_per_employer_and_bulk_skips_existing():
    s = K.CaseStore.sandbox()
    assert _open(s, "PK1") is not None
    assert _open(s, "PK1") is None
    rows = pd.DataFrame({"est_rp": [1e6, 2e6, 3e6], "p_violation": [0.5, 0.6, 0.7], "action": ["a", "b", "c"]},
                        index=["PK1", "PK2", "PK3"])
    assert s.bulk_open("B", rows) == (2, 1)
    assert s.bulk_open("B", rows) == (0, 3)
    assert s.count("B") == 3 and s.count("C") == 0  # branches are separate


def test_summary_and_training_labels():
    s = K.CaseStore.sandbox()
    a, b, c = _open(s, "PK1", est=10e6), _open(s, "PK2", est=4e6), _open(s, "PK3")
    s.close(a, outcome="pelanggaran_terbukti", recovered_rp=8e6)
    s.close(b, outcome="tidak_terbukti", recovered_rp=0)
    sm = s.summary("B")
    assert (sm["total"], sm["open"], sm["closed"], sm["confirmed"], sm["false_alarms"]) == (3, 1, 2, 1, 1)
    assert sm["recovered_rp"] == 8e6 and sm["estimated_rp_closed"] == 14e6 and sm["hit_rate"] == 0.5
    L = s.export_labels("B")
    assert list(L.employer_id) == ["PK1", "PK2"] and list(L.label_violation) == [1, 0]
    assert "PK3" not in set(L.employer_id) and c is not None


def test_persistent_store_survives_reopen(tmp_path):
    path = tmp_path / "cases.sqlite"
    s = K.CaseStore.persistent(path)
    _open(s)
    s.conn.close()
    again = K.CaseStore.persistent(path)
    assert again.get("B", "PK1")["status"] == "Baru"


def test_sandboxes_are_isolated():
    one, two = K.CaseStore.sandbox(), K.CaseStore.sandbox()
    _open(one)
    assert two.get("B", "PK1") is None


def test_draft_approval_hook_logs_and_suggests_next_status():
    s = K.CaseStore.sandbox()
    assert s.record_draft_approval("B", "PK1", "surat_kepatuhan", "Surat kepatuhan") is None  # no case yet
    cid = _open(s)
    nxt = s.record_draft_approval("B", "PK1", "surat_kepatuhan", "Surat kepatuhan")
    assert nxt == "Surat dikirim"
    assert "Tidak dikirim oleh sistem" in s.events(cid).detail.iloc[-1]


def test_example_seed_covers_every_stage_and_is_consistent_with_the_answer_key():
    ranked, truth = _ranked(), _truth()
    s = K.CaseStore.sandbox()
    made = K.seed_examples(s, "B", ranked, truth, today=date(2026, 10, 21))
    assert made >= 8
    assert K.seed_examples(s, "B", ranked, truth) == 0  # never seeds twice
    C = s.cases("B")
    assert set(K.STATUSES) <= set(C.status)
    assert set(C.source) == {"contoh"}
    done = C[C.status == "Selesai"]
    assert set(done.outcome) == set(K.OUTCOMES)
    for _, r in done.iterrows():
        assert bool(truth.at[r.employer_id, "is_violator"]) == (r.outcome in K.VIOLATION_OUTCOMES)
    assert (C.opened_at <= "2026-10-21T23:59:59").all()


def test_draft_types_match_the_drafting_module():
    from tepat import drafting
    assert set(K.DRAFT_NEXT_STATUS) == set(drafting.DOC_LABELS)
