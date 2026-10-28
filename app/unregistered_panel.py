"""Second, synthetic registration queue workspace; ranking uses observed inputs only."""
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st


UNREG_LABEL = {"tepat_registration": "TEPAT daftar kedua", "rules_only": "Aturan hukum (TK dan OSS)",
               "oss_only": "OSS saja", "random": "Acak"}


def budget_selection(U: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Keep ranking order and select only complete inspections within the budget."""
    ranked = U.assign(expected_rp=U.p_unregistered * U.est_rp.clip(lower=0))
    return ranked.loc[ranked.hours.cumsum() <= budget]


def _html(value: object) -> str:
    return escape(str(value), quote=True)


def render(*, branch: str, U: pd.DataFrame, UT: pd.DataFrame, sektor: dict, skala: dict,
           results_dir: Path, jt: Callable, rp: Callable, ribu: Callable, angka: Callable,
           kali: Callable, pct_range: Callable, load_csv: Callable) -> None:
    """Render an interactive list and case detail without changing the source ranking."""
    total_h = float(U.hours.sum())
    maximum = max(25, int(total_h * 0.30))
    default = min(maximum, max(25, int(total_h * 0.10)))

    with st.container(key="workspace_unreg_intro"):
        st.markdown('<div class="ws-intro">Usaha yang wajib mendaftar ke BPJS Kesehatan tetapi belum ada di registri. Daftar terpisah dari antrean utama.</div>', unsafe_allow_html=True)
    if branch == "A":
        st.warning("Cabang A dipakai untuk melatih model daftar ini. Angka uji berasal dari cabang B atau C.")

    # Preset buttons update the native slider before it is instantiated on the rerun.
    with st.container(key="workspace_unreg_capacity"):
        st.markdown('<div class="ws-section-title">Anggaran jam pemeriksaan</div>', unsafe_allow_html=True)
        left, right = st.columns([4, 2], vertical_alignment="center")
        with right:
            presets = st.columns(3)
            for column, share in zip(presets, (5, 10, 20)):
                if column.button(f"{share}%", key=f"unreg_preset_{share}", use_container_width=True):
                    st.session_state["unreg_budget"] = min(maximum, max(25, int(total_h * share / 100 / 25) * 25))
        with left:
            budget = st.slider("Anggaran jam pemeriksaan", 25, maximum, default, step=25,
                               key="unreg_budget", help="Bawaan sekitar 10% total jam kandidat, sama seperti pengujian.",
                               label_visibility="collapsed")
        st.caption(f"Bawaan {ribu(default)} jam, sekitar 10% dari {ribu(int(total_h))} jam kandidat.")

    selected = budget_selection(U, budget)
    ranked = U.assign(expected_rp=U.p_unregistered * U.est_rp.clip(lower=0))
    with st.container(key="workspace_unreg_summary"):
        cards = st.columns(4)
        with cards[0]:
            st.metric("Perkiraan iuran 12 bulan", jt(selected.expected_rp.sum()),
                      help="Tertimbang P. UMK sebagai proksi upah.")
        with cards[1]:
            st.metric("Usaha dalam anggaran", f"{len(selected)}", help=f"Dari {len(U)} kandidat")
        with cards[2]:
            st.metric("Perkiraan Rp per jam", jt(selected.expected_rp.sum() / max(1, selected.hours.sum())),
                      help=f"{ribu(int(selected.hours.sum()))} jam terpakai")
        with cards[3]:
            st.metric("Pemeriksaan bersama TK", f"{int(selected.action.str.startswith('Pemeriksaan bersama').sum())}",
                      help="Dengan BPJS Ketenagakerjaan")

    with st.container(key="workspace_unreg_context"):
        st.info("Antrean utama hanya berisi pemberi kerja yang sudah terdaftar di BPJS Kesehatan. Daftar kedua ini memeriksa "
                "pemegang NIB di OSS dan usaha yang terdaftar di BPJS Ketenagakerjaan, tetapi tidak ada di registri BPJS Kesehatan. "
                "Antrean utama dan angka benchmark utama tidak berubah karena daftar ini.")

    if "unreg_pick" in st.session_state and st.session_state["unreg_pick"] not in U.index:
        del st.session_state["unreg_pick"]
    with st.container(key="workspace_unreg_browser"):
        listing, detail = st.columns([1, 1], gap="medium")
        with listing, st.container(key="workspace_unreg_list"):
            st.markdown('<div class="ws-section-title">Usaha untuk ditinjau</div>', unsafe_allow_html=True)
            for nib, row in ranked.head(10).iterrows():
                source = "BPJS TK dan OSS" if row.tk_n > 0 else "OSS saja"
                sector = sektor.get(row.sector, row.sector)
                with st.container(key=f"workspace_unreg_card_{int(row['rank'])}"):
                    st.markdown(
                        f'<div class="ws-business"><span class="ws-rank">{int(row["rank"])}</span>'
                        f'<span class="ws-business-name">{_html(row["name"])}</span>'
                        f'<span class="ws-business-value">{_html(jt(row.expected_rp / row.hours))}/jam</span>'
                        f'<div class="ws-business-meta">{row.p_unregistered:.0%} kemungkinan wajib dan belum terdaftar. '
                        f'{_html(sector)}, {_html(row.kabkota)}. {_html(source)}.</div></div>',
                        unsafe_allow_html=True)
                    if st.button(f"Tinjau {row['name']}", key=f"unreg_card_{int(row['rank'])}",
                                 use_container_width=True):
                        st.session_state["unreg_pick"] = nib
            st.caption(f"Menampilkan 10 teratas dari {ribu(len(U))} kandidat. Semua peringkat tersedia di tabel lengkap.")
        with detail, st.container(key="workspace_unreg_detail"):
            nib = st.selectbox("Pilih usaha", list(U.index), index=0, key="unreg_pick",
                               format_func=lambda n: f"{U.at[n, 'name']} (peringkat {int(U.at[n, 'rank'])})")
            u = ranked.loc[nib]
            st.subheader(u["name"])
            st.caption(f"{sektor.get(u.sector, u.sector)} di {u.kabkota}. "
                       f"Skala {str(skala.get(u.skala_usaha, u.skala_usaha)).lower()}. NIB {nib} (sintetis).")
            st.markdown(f'<div class="ws-action">Tindakan yang disarankan<br><strong>{_html(u.action)}</strong></div>',
                        unsafe_allow_html=True)
            measures = st.columns(3)
            measures[0].metric("Iuran 12 bulan", jt(u.est_rp))
            measures[1].metric("P(wajib, belum)", f"{u.p_unregistered:.0%}")
            measures[2].metric("Waktu pemeriksaan", f"{int(u.hours)} jam")
            st.markdown('<div class="ws-section-title">Alasan peninjauan</div>', unsafe_allow_html=True)
            for index, text in enumerate(u.reasons.split(" | "), 1):
                st.markdown(f'<div class="ws-reason"><span class="ws-reason-number">{index}</span>'
                            f'<span>{_html(text)}<small>Dasar: {_html(u.legal_basis)}</small></span></div>',
                            unsafe_allow_html=True)
            st.caption(f"{int(u.est_workers)} pekerja x 5% x UMK x 12 bulan. Upah usaha yang belum terdaftar "
                       "tidak diketahui, jadi UMK dipakai sebagai proksi konservatif.")

    with st.expander("Tabel lengkap semua kandidat"):
        show = pd.DataFrame({
            "Peringkat": ranked["rank"], "Anggaran": np.where(ranked.index.isin(selected.index), "Terpilih", "Di luar anggaran"),
            "Usaha": ranked["name"], "P(wajib, belum terdaftar)": ranked.p_unregistered.round(2),
            "Rp/jam": (ranked.expected_rp / ranked.hours).map(jt), "Perkiraan pekerja": ranked.est_workers.astype(int),
            "Sumber": np.where(ranked.tk_n > 0, "BPJS TK dan OSS", "OSS saja"), "Kab/Kota": ranked.kabkota,
            "Sektor": ranked.sector.map(sektor), "Skala": ranked.skala_usaha.map(skala), "Tindakan": ranked.action,
            "Alasan utama": ranked.reasons.str.split(" \\| ").str[0]}, index=ranked.index)
        st.dataframe(show, height=420, width="stretch", hide_index=True,
                     column_config={"Peringkat": st.column_config.NumberColumn("#", width="small"),
                                    "Usaha": st.column_config.TextColumn(width="medium"),
                                    "P(wajib, belum terdaftar)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})

    with st.expander("Kunci jawaban (khusus demo sintetis)"):
        st.caption("Tidak tersedia bagi petugas di dunia nyata. Tidak dipakai untuk menentukan peringkat.")
        t = UT.loc[nib]
        st.write({"wajib dan belum terdaftar": bool(t.is_violator), "jenis": t.kind,
                  "pekerja sebenarnya": int(t.true_workers), "Rp dapat dipulihkan (sebenarnya)": rp(t.recoverable_rp_true)})

    if branch in ("B", "C"):
        committed_results = Path(__file__).resolve().parents[1] / "results"
        source = results_dir if (results_dir / "unregistered_summary.csv").is_file() else committed_results
        us = load_csv(str(source / "unregistered_summary.csv"))
        us = us[us.branch == branch].set_index("ranking")
        runs = load_csv(str(source / "unregistered_runs.csv"))
        runs = runs[runs.branch == branch].pivot_table(index="seed", columns="ranking", values="rp_per_hour_juta")
        per_seed = runs.tepat_registration / runs.rules_only
        ratio = us.at["tepat_registration", "rp_h"] / us.at["rules_only", "rp_h"]
        with st.expander("Hasil uji lima seed", expanded=False):
            with st.container(key="workspace_unreg_benchmark"):
                a, b, c = st.columns(3)
                a.metric("Presisi TEPAT", f"{us.at['tepat_registration', 'precision']:.0%}",
                         help=f"Aturan hukum: {us.at['rules_only', 'precision']:.0%}")
                b.metric("Rp per jam TEPAT", jt(us.at["tepat_registration", "rp_h"] * 1e6),
                         help=f"Aturan hukum: {jt(us.at['rules_only', 'rp_h'] * 1e6)}")
                c.metric("Dibanding aturan hukum", kali(ratio), help="Rasio rata-rata Rp per jam lima seed.")
            rows = [{"Metode": UNREG_LABEL.get(k, k),
                     "Presisi": pct_range(r.precision, r.precision_min, r.precision_max),
                     "Rp jt/jam": f"{angka(r.rp_h)} ({angka(r.rp_h_min)} s.d. {angka(r.rp_h_max)})",
                     "Recall": f"{r.recall:.0%}", "Jujur ditandai": f"{r.honest:.1%}".replace(".", ",")}
                    for k, r in us.loc[[k for k in UNREG_LABEL if k in us.index]].iterrows()]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            tie = " Pada seed terburuk hasilnya hanya setara." if per_seed.min() < 1.01 else ""
            st.caption(f"Keunggulan tipis: rata-rata {kali(ratio)} aturan hukum, per seed {angka(per_seed.min(), 2)} s.d. "
                       f"{angka(per_seed.max(), 2)}.{tie} Rentang dalam kurung adalah nilai terendah s.d. tertinggi dari lima seed. "
                       "Recall rendah karena anggaran hanya cukup untuk sebagian kecil kandidat.")
    st.caption("Batasan: komposisi kandidat (berapa yang wajib, dorman, tutup, atau tercakup grup) adalah asumsi simulasi di "
               "config/unregistered.yaml. Tidak terhubung ke OSS atau BPJS Ketenagakerjaan sungguhan. Usaha tanpa NIB tetap tidak "
               "terlihat. Tingkat kematangan: prototipe.")
