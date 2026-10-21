"""Durability checks use temporary databases, never the real local results directory."""
import json
import os
from pathlib import Path
import subprocess
import sys

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_full_lifecycle_survives_a_fresh_python_process(tmp_path):
    path = tmp_path / "cases.sqlite"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    common = """
import hashlib, json, sys
from tepat.cases import CaseStore
from tepat.reports import ReportStore
cases = CaseStore.persistent(sys.argv[1])
reports = ReportStore(cases)
reporter = hashlib.sha256(b'process-demo').hexdigest()
"""
    writer = common + """
ticket = reports.submit('B', reporter, 'PK1', 'tidak_didaftarkan').ticket
reports.review('B', ticket, status='Ditinjau', reviewer='Pemeriksa C', note='Minta klarifikasi')
cid = reports.link_case('B', ticket, est_rp=1, p_violation=0.2, action='Tinjau')
cases.close(cid, outcome='tidak_terbukti', note='Kesimpulan kasus terpisah')
reports.resolve('B', ticket, disposition='penelaahan_selesai', note='Penelaahan administrasi selesai')
cases.conn.close()
print(json.dumps({'ticket': ticket, 'case_id': cid}))
"""
    write = subprocess.run([sys.executable, "-c", writer, str(path)], env=env, text=True,
                           capture_output=True, timeout=60, check=True)
    initial = json.loads(write.stdout)
    reader = common + """
ticket = sys.argv[2]
r = reports.get('B', ticket)
assert r['status'] == 'Selesai' and r['reviewer'] == 'Pemeriksa C'
assert r['resolution_note'] == 'Penelaahan administrasi selesai'
assert 'Minta klarifikasi' in reports.events('B', ticket).detail.tolist()
case = cases.by_id(r['case_id'])
assert case['status'] == 'Selesai' and case['outcome'] == 'tidak_terbukti'
receipt = reports.submit('B', reporter, 'PK1', 'tidak_didaftarkan')
assert receipt.duplicate and receipt.ticket == ticket
reports.reopen('B', ticket, reason='Klarifikasi tambahan')
assert cases.by_id(r['case_id']) == case
print(json.dumps({'ticket': ticket, 'case_id': r['case_id'], 'reports': reports.count('B'),
                  'cases': cases.count('B'), 'duplicate_count': reports.get('B', ticket)['duplicate_count']}))
cases.conn.close()
"""
    read = subprocess.run([sys.executable, "-c", reader, str(path), initial["ticket"]], env=env, text=True,
                          capture_output=True, timeout=60, check=True)
    assert json.loads(read.stdout) == dict(initial, reports=1, cases=1, duplicate_count=1)


def test_persistent_app_respects_results_directory_and_does_not_seed_examples(tmp_path, monkeypatch):
    monkeypatch.setenv("TEPAT_RESULTS", str(tmp_path))
    monkeypatch.setenv("TEPAT_CASES", "persistent")
    monkeypatch.setenv("TEPAT_DEMO", "1")
    app = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not app.exception
    reports = app.session_state["report_store"]
    assert reports.cases.mode == "persistent"
    assert reports.conn is reports.cases.conn
    assert reports.cases.path == tmp_path / "cases.sqlite"
    assert reports.count("B") == reports.cases.count("B") == 0
    app.sidebar.radio(key="view").set_value("Cek kepesertaan saya").run()
    next(b for b in app.button if b.label == "Kirim laporan demo").click().run()
    assert not app.exception and reports.count("B") == 1 and reports.cases.count("B") == 0
    assert set(reports.reports("B").source) == {"pekerja"}
    assert (tmp_path / "cases.sqlite").is_file() and not (tmp_path / "tips.csv").exists()
    app.sidebar.radio(key="view").set_value("Laporan pekerja").run()
    assert not app.exception
    copy = "\n".join(e.value for kind in ("caption", "info") for e in app.get(kind))
    assert "disimpan lokal" in copy and "aplikasi dimulai ulang" in copy and "direset" not in copy
    reports.conn.close()
