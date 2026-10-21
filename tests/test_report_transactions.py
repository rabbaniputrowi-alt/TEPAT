"""Persistent case/report write transactions must not interleave or use stale states."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading

import pytest

from tepat.cases import CaseStore, CaseError
from tepat.reports import ReportStore, ReportError


def receipt(store, reporter):
    return store.submit('B', hashlib.sha256(reporter.encode()).hexdigest(), 'PK1', 'tidak_didaftarkan').ticket


def test_failed_concurrent_link_cannot_roll_back_another_inspectors_case(tmp_path, monkeypatch):
    cases = CaseStore.persistent(tmp_path / 'cases.sqlite')
    reports = ReportStore(cases)
    first, second = receipt(reports, 'first'), receipt(reports, 'second')
    inserted, second_finished = threading.Event(), threading.Event()
    case_event, report_event = cases._event, reports._event

    def pause_first_insert(cid, actor, kind, detail, at=None):
        if kind == 'dibuka' and threading.current_thread().name.endswith('_0'):
            inserted.set()
            # With correct serialization the second writer waits, so release the
            # first after a bounded wait instead of making the test deadlock.
            second_finished.wait(1)
        return case_event(cid, actor, kind, detail, at)

    def reject_second_link(rid, actor, kind, detail, at):
        if kind == 'kasus ditautkan' and rid == reports.get('B', second)['report_id']:
            raise RuntimeError('injected second-link failure')
        return report_event(rid, actor, kind, detail, at)

    monkeypatch.setattr(cases, '_event', pause_first_insert)
    monkeypatch.setattr(reports, '_event', reject_second_link)

    def link(ticket):
        try:
            return reports.link_case('B', ticket, est_rp=1, p_violation=0.2, action='Tinjau')
        finally:
            if ticket == second:
                second_finished.set()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='triage') as pool:
        one = pool.submit(link, first)
        assert inserted.wait(5), 'First writer did not reach case creation'
        two = pool.submit(link, second)
        cid = one.result(timeout=10)
        with pytest.raises(RuntimeError, match='injected'):
            two.result(timeout=10)
    assert cases.count('B') == 1, 'Another thread rolled back a successful case insertion'
    assert cases.by_id(cid)['employer_id'] == 'PK1'
    assert reports.get('B', first)['case_id'] == cid
    assert reports.get('B', second)['case_id'] is None
    assert cases.conn.execute('PRAGMA foreign_key_check').fetchall() == []
    cases.conn.close()


@pytest.mark.parametrize('separate_connections', [False, True])
def test_successful_concurrent_links_reuse_one_case(tmp_path, separate_connections):
    path = tmp_path / 'cases.sqlite'
    cases = CaseStore.persistent(path)
    reports = ReportStore(cases)
    other_cases = CaseStore.persistent(path) if separate_connections else cases
    other = ReportStore(other_cases) if separate_connections else reports
    first, second = receipt(reports, 'one'), receipt(reports, 'two')
    ready = threading.Barrier(2)

    def link(store, ticket):
        ready.wait(timeout=5)
        return store.link_case('B', ticket, est_rp=1, p_violation=0.2, action='Tinjau')

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(link, reports, first)
        two = pool.submit(link, other, second)
        assert one.result(timeout=10) == two.result(timeout=10)
    assert cases.count('B') == 1
    assert reports.get('B', first)['case_id'] == reports.get('B', second)['case_id']
    assert cases.conn.execute('PRAGMA foreign_keys').fetchone()[0] == 1
    assert cases.conn.execute('PRAGMA foreign_key_check').fetchall() == []
    if separate_connections:
        other_cases.conn.close()
    cases.conn.close()


def test_case_validation_error_rolls_back_earlier_changes_in_the_same_call():
    cases = CaseStore.sandbox()
    cid = cases.open_case('B', 'PK1', est_rp=1, p_violation=0.2, action='Tinjau')
    before = cases.by_id(cid)
    with pytest.raises(CaseError):
        cases.update(cid, status='Surat dikirim', owner='Tidak dikenal')
    assert cases.by_id(cid) == before
    assert cases.events(cid).kind.tolist() == ['dibuka']
    cases.conn.close()


@pytest.mark.parametrize('separate_connections', [False, True])
@pytest.mark.parametrize('operation', ['link', 'review'])
def test_concurrent_transition_cannot_reopen_a_resolved_report(tmp_path, monkeypatch, separate_connections, operation):
    path = tmp_path / 'cases.sqlite'
    cases = CaseStore.persistent(path)
    reports = ReportStore(cases)
    other_cases = CaseStore.persistent(path) if separate_connections else cases
    other = ReportStore(other_cases) if separate_connections else reports
    ticket = receipt(reports, 'reporter')
    read_state, resolved = threading.Event(), threading.Event()
    original_get = reports.get

    def pause_after_read(branch, value):
        row = original_get(branch, value)
        if threading.current_thread().name.endswith('_0') and not read_state.is_set():
            read_state.set()
            resolved.wait(1)
        return row

    monkeypatch.setattr(reports, 'get', pause_after_read)

    def transition():
        try:
            if operation == 'link':
                reports.link_case('B', ticket, est_rp=1, p_violation=0.2, action='Tinjau')
            else:
                reports.review('B', ticket, status='Ditinjau', note='Klarifikasi')
        except ReportError:
            # If resolution serialized first, a stale mutation must be rejected.
            pass

    def resolve():
        try:
            other.resolve('B', ticket, disposition='di_luar_lingkup', note='Penelaahan selesai')
        finally:
            resolved.set()

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='triage') as pool:
        first = pool.submit(transition)
        assert read_state.wait(5)
        second = pool.submit(resolve)
        first.result(timeout=10)
        second.result(timeout=10)
    row = reports.get('B', ticket)
    assert row['status'] == 'Selesai', 'A stale writer reopened the report without a reason'
    assert row['resolution'] == 'di_luar_lingkup' and row['resolved_at']
    assert reports.events('B', ticket).kind.iloc[-1] == 'selesai'
    assert cases.conn.execute('PRAGMA foreign_key_check').fetchall() == []
    if separate_connections:
        other_cases.conn.close()
    cases.conn.close()
