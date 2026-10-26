"""Inspector inbox for synthetic worker allegations; no ranking or automatic case decisions."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

from tepat import cases as K
from tepat import reports as P


SOURCES = {"contoh": "Contoh demo", "pekerja": "Pekerja demo"}


def open_inbox(branch: str, employer_id: str | None = None) -> None:
    st.session_state["view"] = "Laporan pekerja"
    st.session_state[f"reports_employer_{branch}"] = employer_id


def limits(store: P.ReportStore) -> None:
    if store.cases.mode == "sandbox":
        st.info("Versi daring: setiap pengunjung mendapat ruang laporan pribadi berisi contoh. Perubahan tidak terlihat "
                "oleh pengunjung lain. Laporan direset saat sesi baru atau aplikasi dimulai ulang, bukan disimpan permanen.")
    else:
        st.caption("Laporan dan riwayat disimpan lokal di cases.sqlite bersama tindak lanjut. Data tetap ada setelah "
                   "aplikasi dimulai ulang pada komputer ini, selama berkas tersebut tetap tersedia.")
    st.caption("Hanya data sintetis. NIK mentah tidak disimpan di repositori laporan; hanya hash pelapor. "
               "Hash bukan anonimisasi produksi. Belum ada autentikasi, kontrol akses, atau enkripsi produksi. "
               "Tidak dikirim ke BPJS atau Mobile JKN.")


def perform(branch: str, ticket: str, action: Callable[[], object], message: str) -> None:
    try:
        action()
    except P.ReportError as error:
        st.session_state[f"reports_error_{branch}_{ticket}"] = str(error)
    else:
        st.session_state[f"reports_flash_{branch}"] = f"{ticket}: {message}"
    st.rerun()


def render(store: P.ReportStore, branch: str, ranked: pd.DataFrame, reg: pd.DataFrame, *,
           open_case: Callable[[str], None], waktu: Callable[[str], str], legacy_path: Path) -> None:
    limits(store)
    if message := st.session_state.pop(f"reports_flash_{branch}", None):
        st.success(message)
    st.warning("Laporan adalah dugaan pekerja, bukan bukti pelanggaran. Triase dan penyelesaian laporan terpisah "
               "dari hasil pemeriksaan kasus. Pengiriman ulang tidak mengubah peringkat atau membuat kasus baru.")
    if legacy_path.is_file():
        st.caption("Berkas tips.csv lama dipertahankan dan tidak diimpor karena tidak memiliki cabang. "
                   "Isinya tidak dihitung di inbox ini. Impor memerlukan konfirmasi cabang secara eksplisit.")
    rows = store.reports(branch)
    with st.container(key="summary_reports"):
        a, b, c, d = st.columns(4)
        a.metric("Tiket cabang ini", str(len(rows)))
        b.metric("Laporan baru", str(int((rows.status == "Baru").sum())))
        c.metric("Belum selesai", str(int((rows.status != P.CLOSED).sum())))
        d.metric("Pengiriman duplikat", str(int(rows.duplicate_count.sum())))
    with st.expander("Filter laporan"):
        f1, f2, f3 = st.columns(3)
        statuses = f1.multiselect("Status laporan", P.STATUSES, default=P.STATUSES, key=f"reports_statuses_{branch}")
        types = f2.multiselect("Jenis laporan", list(P.TYPES), default=list(P.TYPES),
                              format_func=P.TYPES.get, key=f"reports_types_{branch}")
        reviewers = f3.multiselect("Pemeriksa laporan", K.INSPECTORS, default=K.INSPECTORS,
                                   key=f"reports_reviewers_{branch}")
        employer_ids = sorted(set(reg.index) | set(rows.employer_id),
                              key=lambda e: (reg.at[e, "name"] if e in reg.index else e, e))
        employer_key = f"reports_employer_{branch}"
        if st.session_state.get(employer_key) not in [None, *employer_ids]:
            st.session_state[employer_key] = None
        employer = st.selectbox("Perusahaan laporan", [None, *employer_ids], key=employer_key,
                                format_func=lambda e: "Semua perusahaan" if e is None else reg.at[e, "name"] if e in reg.index else e)
    selected = rows[rows.status.isin(statuses) & rows.report_type.isin(types) & rows.reviewer.isin(reviewers)]
    if employer is not None:
        selected = selected[selected.employer_id == employer]
    st.caption(f"Cabang {branch}: {len(selected)} dari {len(rows)} tiket ditampilkan. Hitungan di atas mencakup seluruh cabang.")
    st.dataframe(pd.DataFrame({
        "Tiket": selected.ticket, "Perusahaan": selected.employer_id.map(reg["name"]).fillna(selected.employer_id),
        "Jenis": selected.report_type.map(P.TYPES), "Status": selected.status, "Pemeriksa": selected.reviewer,
        "Diterima": selected.created_at.map(waktu), "Terakhir masuk": selected.last_received_at.map(waktu),
        "Duplikat": selected.duplicate_count, "Kasus": selected.case_id.map(lambda x: str(int(x)) if pd.notna(x) else "-"),
        "Sumber": selected.source.map(SOURCES),
    }), hide_index=True, width="stretch")
    if selected.empty:
        st.info("Belum ada laporan sesuai filter. Pilih filter lain atau kirim laporan dari Cek kepesertaan saya.")
        return
    ticket_key = f"reports_ticket_{branch}"
    tickets = selected.ticket.tolist()
    if st.session_state.get(ticket_key) not in tickets:
        st.session_state[ticket_key] = tickets[0]
    ticket = st.selectbox("Pilih tiket", tickets, key=ticket_key)
    report = store.get(branch, ticket)
    with st.container(border=True, key="glass_rp_1"):
        st.subheader(ticket)
        st.write(f"Jenis: {P.TYPES[report['report_type']]}. Status: {report['status']}. Pemeriksa: {report['reviewer']}.")
        st.caption(f"Diterima {waktu(report['created_at'])}. Terakhir masuk {waktu(report['last_received_at'])}. "
                   f"Duplikat {report['duplicate_count']}. Sumber: {SOURCES[report['source']]}.")
        if report["source"] == "contoh":
            st.caption("Contoh demo untuk latihan triase, bukan laporan pekerja sungguhan. Tidak memakai kunci jawaban pemeriksaan.")
        st.button("Buka perusahaan", key=f"reports_open_{branch}_{ticket}", on_click=open_case,
                  args=(report["employer_id"],), disabled=report["employer_id"] not in ranked.index)
    events = store.events(branch, ticket)
    # Every editable widget includes the ticket and current history length. A rerun after
    # a write must mount a fresh form, not silently restore the browser's previous values.
    revision = f"{branch}_{ticket}_{len(events)}"
    problem = st.session_state.pop(f"reports_error_{branch}_{ticket}", None)
    if problem:
        st.error(problem)
    if report["status"] != P.CLOSED:
        with st.form(key=f"reports_review_form_{revision}"):
            first, second = st.columns(2)
            available = P.OPEN_STATUSES if report["case_id"] is not None else P.OPEN_STATUSES[:2]
            status = first.selectbox("Status triase", available, index=available.index(report["status"]),
                                     key=f"reports_status_{revision}")
            reviewer = second.selectbox("Pemeriksa penelaah", K.INSPECTORS,
                                        index=K.INSPECTORS.index(report["reviewer"]), key=f"reports_reviewer_{revision}")
            note = st.text_area("Catatan investigasi", key=f"reports_note_{revision}", height=100,
                                help="Catat klarifikasi dan dokumen yang ditinjau, tanpa NIK atau data nyata.")
            if st.form_submit_button("Simpan penelaahan"):
                perform(branch, ticket, lambda: store.review(branch, ticket, status=status, reviewer=reviewer, note=note),
                        "Penelaahan tersimpan. Status kasus tidak diubah.")
    with st.container(border=True, key="glass_rp_2"):
        st.subheader("Tindak lanjut perusahaan")
        case = store.cases.get(branch, report["employer_id"])
        if report["case_id"] is not None:
            st.write(f"Tertaut ke kasus {report['case_id']}. Status kasus: {case['status']}.")
            st.caption("Penyelesaian laporan tidak menutup kasus. Kasus selesai tidak dibuka kembali oleh penautan laporan.")
        elif report["status"] != P.CLOSED:
            if case:
                st.caption(f"Kasus {case['case_id']} sudah ada dengan status {case['status']}. "
                           "Gunakan kasus yang sama tanpa mengubah pemeriksa, jadwal, atau hasilnya.")
            else:
                st.caption("Belum ada kasus. Pembuatan tindak lanjut memerlukan keputusan petugas; tidak ada surat yang dikirim.")
            if report["employer_id"] in ranked.index:
                r = ranked.loc[report["employer_id"]]
                if st.button("Tautkan kasus yang ada" if case else "Buat dan tautkan tindak lanjut",
                             key=f"reports_link_{revision}"):
                    perform(branch, ticket, lambda: store.link_case(branch, ticket, est_rp=r.est_rp,
                            p_violation=r.p_violation, action=r.action),
                            "Kasus ditautkan dan dicatat di kedua riwayat. Hasil pemeriksaan tidak diubah.")
            else:
                st.info("Perusahaan tidak tersedia di data cabang ini. Penautan kasus belum dapat dilakukan.")
        else:
            st.caption("Laporan selesai tanpa tautan kasus. Buka kembali laporan dengan alasan bila perlu ditindaklanjuti.")
    if report["status"] != P.CLOSED:
        with st.expander("Selesaikan triase laporan", expanded=bool(problem)):
            st.caption("Disposisi menutup penelaahan laporan, bukan menetapkan pelanggaran atau menutup kasus perusahaan.")
            with st.form(key=f"reports_resolve_form_{revision}"):
                disposition = st.radio("Disposisi laporan", list(P.DISPOSITIONS), index=None,
                                       format_func=P.DISPOSITIONS.get, key=f"reports_disposition_{revision}")
                resolution_note = st.text_area("Catatan penyelesaian", key=f"reports_resolution_note_{revision}")
                if st.form_submit_button("Selesaikan laporan"):
                    perform(branch, ticket, lambda: store.resolve(branch, ticket, disposition=disposition, note=resolution_note),
                            "Laporan selesai. Status dan hasil kasus tetap terpisah.")
    else:
        st.markdown(f"**Disposisi:** {P.DISPOSITIONS[report['resolution']]}. Selesai {waktu(report['resolved_at'])}.")
        st.write(report["resolution_note"])
        with st.expander("Buka kembali triase", expanded=bool(problem)):
            with st.form(key=f"reports_reopen_form_{revision}"):
                reason = st.text_input("Alasan membuka kembali laporan", key=f"reports_reopen_reason_{revision}")
                if st.form_submit_button("Buka kembali laporan"):
                    perform(branch, ticket, lambda: store.reopen(branch, ticket, reason=reason),
                            "Laporan dibuka kembali. Kasus perusahaan tidak diubah.")
    st.subheader("Riwayat laporan")
    st.dataframe(pd.DataFrame({"Waktu": events["at"].map(waktu), "Oleh": events.actor,
                               "Jenis": events.kind, "Keterangan": events.detail}), hide_index=True, width="stretch")
