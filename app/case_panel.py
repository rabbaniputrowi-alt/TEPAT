"""Observed-evidence case workspace. The parent owns employer selection and heading."""
from __future__ import annotations

from datetime import datetime
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from tepat import cases as K, drafting, scorer
import drafting_panel


_COMPONENTS = (
    ("Pekerja tidak terdaftar", "rp_headcount", True),
    ("Upah (TK > Kes)", "rp_wage2c", True),
    ("Upah (pola sejenis)", "rp_wage_peer", True),
    ("Tunggakan iuran", "rp_arrears", True),
    ("Terlambat daftar", "rp_lag", True),
    ("Nonaktif sebelum keluar", "rp_onoff", True),
    ("Pindah segmen", "rp_group_move", True),
    ("Klinik perusahaan", "rp_clinic", True),
    ("PHK berulang", "rp_phk_gap", True),
    ("Biaya rawat saat nonaktif", "rp_care_liability", False),
    ("Tanggungan PHK", "rp_phk_cover", False),
)


def composition(features: pd.Series, lookback: int) -> pd.DataFrame:
    """Positive estimated recovery components, using the scorer's lookback convention."""
    return pd.DataFrame(
        [(label, amount) for label, field, recurring in _COMPONENTS
         if (amount := float(features[field]) * (lookback if recurring else 1)) > 0],
        columns=["Komponen", "Rp"],
    )


def _follow_up(store, branch, eid, ranked_row, jt, tanggal, waktu, flash):
    case = store.get(branch, eid)
    with st.container(border=True, key="ws_case_follow_up"):
        st.subheader("Tindak lanjut")
        if msg := st.session_state.pop("fu_flash", None):
            st.success(msg)
        problem = st.session_state.pop("fu_error", None)
        if problem:
            st.error(problem)
        if case is None:
            st.caption("Belum ada tindak lanjut untuk perusahaan ini. Mulai bila petugas memutuskan untuk menindaklanjuti.")
            if st.button("Mulai tindak lanjut", key="fu_start", type="primary"):
                store.open_case(branch, eid, est_rp=ranked_row.est_rp,
                                p_violation=ranked_row.p_violation, action=ranked_row.action)
                flash("Tindak lanjut dimulai dengan status Baru.")
                st.rerun()
            return
        cid = int(case["case_id"])
        v = f"{branch}_{cid}_{len(store.events(cid))}"
        line = f"Status **{case['status']}**. {case['owner']}."
        if case["status"] == K.CLOSED:
            line += (f" Hasil: {K.OUTCOMES[case['outcome']]}, Rp dipulihkan {jt(case['recovered_rp'] or 0)}."
                     f" Selesai {tanggal(case['closed_at'])}.")
        else:
            late = case["due_date"] and case["due_date"] < datetime.now().date().isoformat()
            line += f" Tindak lanjut berikutnya {tanggal(case['due_date'])}" + (" (lewat jadwal)." if late else ".")
        st.markdown(line)
        sug = st.session_state.get("fu_suggest")
        if sug and sug[:2] == (branch, eid) and sug[3] and case["status"] != K.CLOSED and sug[3] != case["status"]:
            s1, s2 = st.columns([3, 1], gap="large", vertical_alignment="center")
            s1.info(f"{sug[2]} sudah disetujui. Saran status berikutnya: {sug[3]}. Sistem tidak mengirim apa pun; "
                    "ubah status setelah dokumen benar-benar dikirim atau pemeriksaan dijadwalkan.")
            if s2.button(f"Ubah ke {sug[3]}", key="fu_apply_suggest", width="stretch"):
                store.update(cid, status=sug[3])
                st.session_state.pop("fu_suggest", None)
                flash(f"Status diubah ke {sug[3]}.")
                st.rerun()
        if case["source"] == "contoh":
            st.caption("Kasus contoh untuk demo. Hasil kasus contoh yang sudah selesai diambil dari kunci jawaban sintetis.")
        if case["status"] != K.CLOSED:
            with st.form(key=f"fu_update_form_{v}"):
                c1, c2, c3 = st.columns(3)
                status = c1.selectbox("Status", K.OPEN_STATUSES, index=K.OPEN_STATUSES.index(case["status"]), key=f"fu_status_{v}")
                owner = c2.selectbox("Pemeriksa", K.INSPECTORS, index=K.INSPECTORS.index(case["owner"]), key=f"fu_owner_{v}")
                due = c3.date_input("Tanggal tindak lanjut", value=datetime.fromisoformat(case["due_date"]).date()
                                    if case["due_date"] else None, format="DD/MM/YYYY", key=f"fu_due_{v}")
                note = st.text_area("Catatan", key=f"fu_note_{v}", height=80,
                                    placeholder="Misalnya hasil telepon ke bagian personalia")
                if st.form_submit_button("Simpan perubahan"):
                    changes = store.update(cid, status=status, owner=owner, due=due, note=note)
                    flash("Perubahan tersimpan di riwayat." if changes else "Tidak ada perubahan.")
                    st.rerun()
            with st.expander("Selesaikan kasus dengan hasil", expanded=bool(problem)):
                with st.form(key=f"fu_close_form_{v}"):
                    outcome = st.radio("Hasil pemeriksaan", list(K.OUTCOMES), index=None, format_func=K.OUTCOMES.get,
                                       key=f"fu_outcome_{v}")
                    recovered = st.number_input("Rp dipulihkan (0 bila belum ada)", min_value=0, value=0, step=100_000,
                                                key=f"fu_recovered_{v}")
                    close_note = st.text_input("Catatan penutup", key=f"fu_close_note_{v}")
                    if st.form_submit_button("Tutup kasus"):
                        try:
                            store.close(cid, outcome=outcome, recovered_rp=float(recovered), note=close_note)
                        except K.CaseError as exc:
                            st.session_state["fu_error"] = str(exc)
                            st.rerun()
                        else:
                            flash("Kasus selesai. Hasilnya masuk ringkasan di halaman Tindak lanjut.")
                            st.rerun()
        else:
            with st.expander("Buka kembali kasus", expanded=bool(problem)):
                with st.form(key=f"fu_reopen_form_{v}"):
                    reason = st.text_input("Alasan membuka kembali", key=f"fu_reopen_reason_{v}")
                    if st.form_submit_button("Buka kembali"):
                        try:
                            store.reopen(cid, reason=reason)
                        except K.CaseError as exc:
                            st.session_state["fu_error"] = str(exc)
                            st.rerun()
                        else:
                            flash("Kasus dibuka kembali dengan status Menunggu koreksi.")
                            st.rerun()
        with st.expander("Riwayat tindak lanjut"):
            ev = store.events(cid)
            st.dataframe(pd.DataFrame({"Waktu": ev["at"].map(waktu), "Oleh": ev.actor,
                                       "Jenis": ev.kind, "Keterangan": ev.detail}),
                         hide_index=True, width="stretch")


def render(*, branch, eid, ranked, F, reg, obs, store, reports, branch_dir, sektor,
           chart_config, open_case, open_inbox, load_truth, jt, rp, tanggal, waktu, flash, lookback):
    """Render the selected employer below the caller's title and employer selector."""
    r, f = ranked.loc[eid], F.loc[eid]
    with st.container(key="ws_case_summary"):
        a, b, c, d = st.columns(4)
        a.metric("Perkiraan pemulihan", jt(r.est_rp),
                 help=f"Rentang {jt(r.rp_low)} s.d. {jt(r.rp_high)} (persentil 10-90 dari data latih)")
        b.metric("Peringkat", f"#{int(r['rank'])}")
        c.metric("P(pelanggaran)", f"{r.p_violation:.0%}")
        d.metric("Waktu pemeriksaan", f"{int(f.hours)} jam")
    n_reports = reports.count(branch, eid)
    with st.container(key="ws_case_recommended"):
        lead, action = st.columns([4, 1], vertical_alignment="center")
        with lead:
            st.markdown("**Tindakan yang disarankan**")
            st.markdown(escape(str(r.action)))
            if r.pay_label:
                st.caption(f"Catatan pembayaran: {r.pay_label}")
            st.caption(f"{n_reports} tiket laporan pekerja di cabang ini untuk perusahaan ini. "
                       "Laporan masih dugaan dan tidak mengubah peringkat.")
        action.button("Lihat laporan pekerja", key="case_reports", on_click=open_inbox, args=(branch, eid))
    left, right = st.columns([3, 2], gap="large")
    with left:
        _follow_up(store, branch, eid, r, jt, tanggal, waktu, flash)
        with st.container(border=True, key="ws_case_reasons"):
            st.subheader("Mengapa perusahaan ini?")
            reasons = scorer.reasons(f)
            if not reasons:
                st.success("Tidak ada sinyal pelanggaran. Jalur hijau.")
            for _, text, basis in reasons[:6]:
                st.markdown(f'<div class="signal-row ws-signal-row"><strong>{escape(str(text))}</strong>'
                            f'<small>Dasar: {escape(str(basis))}</small></div>', unsafe_allow_html=True)
        with st.container(border=True, key="ws_case_documents"):
            st.subheader("Dokumen pemeriksaan")
            documents = "".join(f"<li>{escape(str(doc))}</li>" for doc in scorer.brief(f))
            st.markdown(f'<ul class="document-list ws-document-list">{documents}</ul>', unsafe_allow_html=True)
        def on_draft_approved(employer_id: str, doc_type: str, text: str) -> None:
            st.session_state.setdefault("approved_drafts", {})[f"{employer_id}:{doc_type}"] = {
                "employer_id": employer_id, "doc_type": doc_type, "text": text,
                "approved_at": datetime.now().isoformat(timespec="seconds")}
            label = drafting.DOC_LABELS[doc_type]
            if store.get(branch, employer_id) is None:
                st.info("Persetujuan belum tercatat di riwayat karena perusahaan ini belum punya tindak lanjut. "
                        "Tekan Mulai tindak lanjut di atas, lalu setujui draf lagi.")
                return
            nxt = store.record_draft_approval(branch, employer_id, doc_type, label)
            st.session_state["fu_suggest"] = (branch, employer_id, label, nxt)
            flash(f"{label} disetujui dan dicatat di riwayat tindak lanjut.")
            st.rerun()
        drafting_panel.render(eid, r, f, reg.loc[eid], sektor[reg.at[eid, "sector"]], on_approved=on_draft_approved)
    with right:
        with st.container(border=True, key="ws_case_composition"):
            st.subheader("Komposisi perkiraan Rp")
            comp = composition(f, lookback)
            if not comp.empty:
                st.altair_chart(alt.Chart(comp).mark_bar(color="#087f82", cornerRadiusEnd=4, height=20).encode(
                    x=alt.X("Rp:Q", title="Rp (12 bulan)", axis=alt.Axis(format="~s")),
                    y=alt.Y("Komponen:N", sort="-x", title=None),
                    tooltip=["Komponen", alt.Tooltip("Rp:Q", format=",.0f")]
                ).properties(height=max(130, len(comp) * 32)).configure(**chart_config), theme=None, width="stretch")
            else:
                st.caption("Tidak ada komponen Rp yang teridentifikasi.")
        with st.container(border=True, key="ws_case_payments"):
            st.subheader("Pembayaran iuran 24 bulan")
            bill = obs["billing"][obs["billing"].employer_id == eid].copy()
            if bill.empty:
                st.caption("Belum ada riwayat pembayaran iuran untuk perusahaan ini.")
            else:
                month_names = ("Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des")
                bill["Bulan"] = bill.month.map(lambda m: f"{month_names[(9 + int(m)) % 12]} {2024 + (9 + int(m)) // 12}")
                bill["Badan"] = bill.agency.map({"KS": "BPJS Kesehatan", "TK": "BPJS Ketenagakerjaan"})
                bill["Terlambat (hari)"] = bill.days_late
                bill["Kurang bayar"] = (bill.billed_rp - bill.paid_rp).clip(lower=0)
                st.altair_chart(alt.Chart(bill).mark_line(point=True, strokeWidth=2.5).encode(
                    x=alt.X("month:Q", title="Bulan ke- (0 = Okt 2024)", axis=alt.Axis(tickCount=6)),
                    y=alt.Y("Terlambat (hari):Q", title="Hari keterlambatan"),
                    color=alt.Color("Badan:N", scale=alt.Scale(range=["#087f82", "#d48b4a"]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Bulan", "Badan", "Terlambat (hari)", alt.Tooltip("Kurang bayar:Q", format=",.0f")]
                ).properties(height=210).configure(**chart_config), theme=None, width="stretch")
    with st.expander("Kunci jawaban (khusus demo sintetis)"):
        st.caption("Tidak tersedia bagi petugas di dunia nyata. Tidak dipakai untuk menentukan peringkat.")
        t = load_truth(str(branch_dir)).loc[eid]
        st.write({"melanggar": bool(t.is_violator), "varian yang ditanam": t.variants or "-",
                  "look-alike jujur": t.lookalikes or "-", "Rp dapat dipulihkan (sebenarnya)": rp(t.recoverable_rp_true)})
