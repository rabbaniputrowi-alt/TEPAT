"""TEPAT demo app (synthetic data only).

    uv run streamlit run app/streamlit_app.py

Inspector views read observed/ + the scorer's ranked output. Only the clearly labelled
"kunci jawaban" expanders and the benchmark tab read truth/ (to show how the method was scored).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from tepat import rules as R
from tepat import scorer

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TEPAT_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("TEPAT_RESULTS", ROOT / "results"))
TIPS = RESULTS / "tips.csv"
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
SEKTOR = {"C": "Manufaktur", "F": "Konstruksi", "G": "Perdagangan", "H": "Logistik", "I": "Akomodasi & Makan",
          "J": "Informasi & Kom.", "K": "Keuangan", "Q": "Kesehatan", "A": "Pertanian"}
RANK_LABEL = {"random": "Acak", "biggest_first": "Terbesar dulu", "rules_only": "Aturan saja",
              "tepat_v1": "TEPAT v1", "tepat_v2": "TEPAT v2", "tepat_v2_1": "TEPAT v2.1",
              "tepat_v2_1_with_random_arm": "TEPAT v2.1 + jalur acak 8%"}

st.set_page_config(page_title="TEPAT · Pemeriksaan Kepatuhan Tepat Sasaran", page_icon="🎯", layout="wide")


# --------------------------------------------------------------------------- helpers

def rp(x: float) -> str:
    return "Rp" + f"{x:,.0f}".replace(",", ".")


def jt(x: float) -> str:
    return "Rp" + f"{x / 1e6:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".") + " jt"


def month_label(m: int) -> str:
    return f"{BULAN[(9 + m) % 12]} {2024 + (9 + m) // 12}"


def day_label(d: float | None) -> str:
    if d is None or pd.isna(d):
        return "-"
    d = int(d)
    if d < 0:
        return "sebelum Okt 2024"
    return f"{d % 30 + 1} {month_label(d // 30)}"


@st.cache_data(show_spinner="Memuat data sintetis…")
def load(branch: str):
    obs = scorer.load_observed(DATA / branch)
    ranked = pd.read_parquet(RESULTS / f"ranked_{branch}.parquet")
    F = pd.read_parquet(RESULTS / f"features_{branch}.parquet")
    reg = obs["employer_registry"].set_index("employer_id")
    return obs, ranked, F, reg


@st.cache_data
def load_truth(branch: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / branch / "truth" / "employers_truth.parquet").set_index("employer_id")


@st.cache_data
def load_results() -> dict:
    return json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))


def read_tips() -> pd.DataFrame:
    if TIPS.exists():
        return pd.read_csv(TIPS)
    return pd.DataFrame(columns=["waktu", "tiket", "nik_hash", "employer_id", "jenis"])


def add_tip(nik: str, employer_id: str, jenis: str) -> str:
    TIPS.parent.mkdir(parents=True, exist_ok=True)
    tiket = "TPT-" + hashlib.sha1(f"{nik}{employer_id}{jenis}{datetime.now()}".encode()).hexdigest()[:6].upper()
    new = not TIPS.exists()
    with open(TIPS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["waktu", "tiket", "nik_hash", "employer_id", "jenis"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), tiket,
                    hashlib.sha256(nik.encode()).hexdigest()[:12], employer_id, jenis])
    return tiket


# --------------------------------------------------------------------------- sidebar

st.sidebar.title("🎯 TEPAT")
st.sidebar.caption("Pemeriksaan Kepatuhan Tepat Sasaran · Tim Johannes Kepler · Healthkathon 2026")
branch = st.sidebar.selectbox("Cabang data sintetis", ["B", "C", "A"], index=0,
                              format_func=lambda b: {"A": "A (tuning)", "B": "B (uji)", "C": "C (uji bergeser)"}[b])
st.sidebar.warning("Semua perusahaan, pekerja, NIK dan angka di sini **sintetis**. NIK diawali 99 (bukan kode provinsi). "
                   "UMK memakai kab/kota fiktif.", icon="⚠️")

obs, ranked, F, reg = load(branch)
lookback = 12
D = R.SCORING_DAY

tab_list, tab_case, tab_bench, tab_fair, tab_worker = st.tabs(
    ["📋 Daftar Prioritas", "🔎 Kasus Perusahaan", "📊 Benchmark", "⚖️ Keadilan & Jalur Acak", "🙋 Cek Kepesertaan Saya"])

# --------------------------------------------------------------------------- 1. inspector list

with tab_list:
    st.subheader("Daftar pemeriksaan, diurutkan menurut perkiraan Rp yang dipulihkan per jam pemeriksaan")
    L = ranked.join(reg[["name", "sector", "kabkota"]]).join(F[["band_obs", "hours"]])
    c1, c2, c3, c4 = st.columns(4)
    kab = c1.multiselect("Kab/Kota", sorted(L.kabkota.unique()), default=sorted(L.kabkota.unique()))
    tiers = c2.multiselect("Tingkat", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang"])
    queue = c3.radio("Antrean", ["inspeksi", "cicilan"], horizontal=True,
                     format_func=lambda q: {"inspeksi": "Pemeriksaan", "cicilan": "Cicilan (tidak mampu bayar)"}[q])
    total_h = float(F.hours.sum())
    budget = c4.slider("Anggaran jam pemeriksaan", 50, int(total_h * 0.3), int(total_h * 0.10), step=25)
    insp = L[L.queue == "inspeksi"]
    in_budget = set(insp.index[insp.hours.cumsum() <= budget])
    V = L[L.kabkota.isin(kab) & L.tier.isin(tiers) & (L.queue == queue)].copy()
    V["anggaran"] = np.where(V.index.isin(in_budget), "✅", "")
    m1, m2, m3, m4 = st.columns(4)
    sel = L.loc[list(in_budget)]
    m1.metric("Perusahaan dalam anggaran", f"{len(sel)}")
    m2.metric("Perkiraan Rp dipulihkan", jt(sel.expected_rp.sum()))
    m3.metric("Rp per jam (perkiraan)", jt(sel.expected_rp.sum() / max(1, sel.hours.sum())))
    m4.metric("Antrean cicilan", f"{int((L.queue == 'cicilan').sum())} perusahaan")
    show = pd.DataFrame({
        "Peringkat": V["rank"], "Anggaran": V.anggaran, "Perusahaan": V["name"], "Sektor": V.sector.map(SEKTOR),
        "Kab/Kota": V.kabkota, "Skala": V.band_obs, "Tingkat": V.tier,
        "P(pelanggaran)": V.p_violation.round(2), "Perkiraan Rp": V.est_rp.map(jt),
        "Rentang": [f"{jt(a)} – {jt(b)}" for a, b in zip(V.rp_low, V.rp_high)],
        "Rp/jam": (V.expected_rp / V.hours).map(jt), "Alasan utama": V.top_reasons.str.split(" \\| ").str[0],
        "Tindakan": V.action}, index=V.index)
    st.dataframe(show, height=520, width="stretch",
                 column_config={"P(pelanggaran)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
    st.caption("Keputusan memeriksa tetap di tangan petugas. TEPAT hanya mengurutkan dan menjelaskan (UU PDP Ps. 10: tidak ada "
               "keputusan otomatis semata). 8% anggaran disisihkan untuk pemeriksaan acak (lihat tab Keadilan).")

# --------------------------------------------------------------------------- 2. case page

with tab_case:
    tips = read_tips()
    order = list(ranked.index)
    default = order[0]
    opts = order[:300]
    eid = st.selectbox("Pilih perusahaan", opts, index=0,
                       format_func=lambda e: f"#{int(ranked.at[e, 'rank'])} · {reg.at[e, 'name']} ({e})")
    r, f_ = ranked.loc[eid], F.loc[eid]
    st.subheader(f"{reg.at[eid, 'name']} · {SEKTOR[reg.at[eid, 'sector']]} · {reg.at[eid, 'kabkota']}")
    a, b, c, d = st.columns(4)
    a.metric("Peringkat", f"#{int(r['rank'])} ({r.tier})")
    b.metric("P(pelanggaran)", f"{r.p_violation:.0%}")
    c.metric("Perkiraan Rp dapat dipulihkan", jt(r.est_rp), help=f"Rentang {jt(r.rp_low)} – {jt(r.rp_high)} (persentil 10–90 dari data latih)")
    d.metric("Perkiraan jam pemeriksaan", f"{int(f_.hours)} jam")
    st.info(f"**Tindakan yang disarankan:** {r.action}" + (f"  \n**Catatan pembayaran:** {r.pay_label}" if r.pay_label else ""))
    n_tip = int((tips.employer_id == eid).sum()) if len(tips) else 0
    if n_tip:
        st.warning(f"📣 {n_tip} laporan pekerja masuk melalui *Cek Kepesertaan Saya* untuk perusahaan ini.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Mengapa perusahaan ini?")
        rs = scorer.reasons(f_)
        if not rs:
            st.success("Tidak ada sinyal pelanggaran. Jalur hijau.")
        for _, text, basis in rs[:6]:
            st.markdown(f"- {text}  \n  <span style='opacity:.7'>Dasar: {basis}</span>", unsafe_allow_html=True)
        st.markdown("#### Dokumen yang diminta saat pemeriksaan")
        for x in scorer.brief(f_):
            st.markdown(f"- {x}")
    with right:
        st.markdown("#### Komposisi perkiraan Rp")
        comp = pd.DataFrame({"Komponen": ["Pekerja tidak terdaftar", "Upah (TK > Kes)", "Upah (pola sejenis)", "Tunggakan iuran",
                                          "Terlambat daftar", "Nonaktif sebelum keluar", "Pindah segmen", "Klinik perusahaan",
                                          "PHK berulang", "Biaya rawat saat nonaktif", "Tanggungan PHK"],
                             "Rp": [f_.rp_headcount * lookback, f_.rp_wage2c * lookback, f_.rp_wage_peer * lookback,
                                    f_.rp_arrears * lookback, f_.rp_lag * lookback, f_.rp_onoff * lookback,
                                    f_.rp_group_move * lookback, f_.rp_clinic * lookback, f_.rp_phk_gap * lookback,
                                    f_.rp_care_liability, f_.rp_phk_cover]})
        comp = comp[comp.Rp > 0]
        if len(comp):
            st.altair_chart(alt.Chart(comp).mark_bar().encode(
                x=alt.X("Rp:Q", title="Rp (12 bulan)"), y=alt.Y("Komponen:N", sort="-x", title=None),
                tooltip=["Komponen", alt.Tooltip("Rp:Q", format=",.0f")]), width="stretch")
        else:
            st.caption("Tidak ada komponen Rp.")
        bill = obs["billing"][obs["billing"].employer_id == eid].copy()
        bill["Bulan"] = bill.month.map(month_label)
        bill["Badan"] = bill.agency.map({"KS": "BPJS Kesehatan", "TK": "BPJS Ketenagakerjaan"})
        bill["Terlambat (hari)"] = bill.days_late
        bill["Kurang bayar"] = (bill.billed_rp - bill.paid_rp).clip(lower=0)
        st.markdown("#### Pembayaran iuran 24 bulan")
        st.altair_chart(alt.Chart(bill).mark_line(point=True).encode(
            x=alt.X("month:Q", title="Bulan ke- (0 = Okt 2024)"), y=alt.Y("Terlambat (hari):Q"),
            color=alt.Color("Badan:N", legend=alt.Legend(orient="bottom")),
            tooltip=["Bulan", "Badan", "Terlambat (hari)", alt.Tooltip("Kurang bayar:Q", format=",.0f")]).properties(height=200),
            width="stretch")
    with st.expander("🔑 Kunci jawaban (hanya untuk demo sintetis, tidak tersedia di dunia nyata)"):
        T = load_truth(branch)
        t = T.loc[eid]
        st.write({"melanggar": bool(t.is_violator), "varian yang ditanam": t.variants or "-",
                  "look-alike jujur": t.lookalikes or "-", "Rp dapat dipulihkan (sebenarnya)": rp(t.recoverable_rp_true)})

# --------------------------------------------------------------------------- 3. benchmark

with tab_bench:
    res = load_results()
    rb = res["branches"][branch]
    st.subheader(f"Cabang {branch}: anggaran {rb['budget_hours']:.0f} jam (≈10% total jam), "
                 f"{rb['violators']} pelanggar dari {rb['employers']} perusahaan ({rb['prevalence']:.0%})")
    if branch == "A":
        st.warning("Cabang A dipakai untuk tuning. Angka utama selalu dari cabang B atau C.")
    rows = []
    for k, v in rb["rankings"].items():
        rows.append(dict(Peringkat=RANK_LABEL.get(k, k), key=k, rp_per_jam=v["rp_per_hour"] / 1e6, presisi=v["precision"],
                         recall=v["recall"], jujur_ditandai=v["honest_flag_rate"], diperiksa=v["inspected"],
                         rp_total=v["rp_recovered"] / 1e6))
    B = pd.DataFrame(rows)
    B["kelompok"] = np.where(B.key.str.startswith("tepat"), "TEPAT", "Pembanding")
    x1, x2 = st.columns(2)
    x1.altair_chart(alt.Chart(B).mark_bar().encode(
        x=alt.X("rp_per_jam:Q", title="Rp (juta) dipulihkan per jam pemeriksaan"),
        y=alt.Y("Peringkat:N", sort="-x", title=None), color=alt.Color("kelompok:N", legend=None),
        tooltip=["Peringkat", alt.Tooltip("rp_per_jam:Q", format=".2f"), "diperiksa"]).properties(title="Rp per jam"),
        width="stretch")
    x2.altair_chart(alt.Chart(B).mark_bar().encode(
        x=alt.X("presisi:Q", title="Presisi (bagian yang diperiksa memang melanggar)", axis=alt.Axis(format="%")),
        y=alt.Y("Peringkat:N", sort="-x", title=None), color=alt.Color("kelompok:N", legend=None),
        tooltip=["Peringkat", alt.Tooltip("presisi:Q", format=".0%")]).properties(title="Presisi"),
        width="stretch")
    st.dataframe(B.drop(columns=["key", "kelompok"]).style.format(
        {"rp_per_jam": "{:.2f}", "presisi": "{:.0%}", "recall": "{:.0%}", "jujur_ditandai": "{:.1%}",
         "diperiksa": "{:.0f}", "rp_total": "{:,.0f}"}), width="stretch", hide_index=True)
    final = rb.get("final_ranking", "tepat_v2_1")
    y1, y2 = st.columns(2)
    with y1:
        st.markdown("**Recall per varian pelanggaran** (di mana TEPAT masih lemah)")
        rv = pd.DataFrame({"Aturan saja": rb["rankings"]["rules_only"]["recall_by_variant"],
                           "TEPAT": rb["rankings"][final]["recall_by_variant"]})
        st.dataframe(rv.style.format("{:.0%}"), width="stretch")
    with y2:
        st.markdown("**Look-alike jujur yang ikut ditandai** (lebih rendah lebih baik)")
        lv = pd.DataFrame({"Aturan saja": rb["rankings"]["rules_only"]["lookalike_flag_rate"],
                           "TEPAT": rb["rankings"][final]["lookalike_flag_rate"]})
        st.dataframe(lv.style.format("{:.0%}"), width="stretch")
    cov = rb.get("rp_range_coverage")
    if cov is not None:
        st.caption(f"Kalibrasi rentang Rp: {cov:.0%} pelanggar yang terpilih punya Rp sebenarnya di dalam rentang yang dinyatakan "
                   "(target ≈80%, persentil 10–90).")
    st.markdown("#### Contoh: tidak ditandai, dan memang benar")
    T = load_truth(branch)
    rules = scorer.rules_only(F)
    honest = T[~T.is_violator & T.lookalikes.str.contains("C1")].index
    ex = [e for e in rules.index[:200] if e in honest]
    if ex:
        e = ex[0]
        st.success(f"**{reg.at[e, 'name']}** (konstruksi): aturan sederhana menandainya (TK > Kes karena pekerja proyek), "
                   f"peringkat aturan #{list(rules.index).index(e) + 1}. TEPAT mengenali program `project` di data TK dan "
                   f"menempatkannya di #{int(ranked.at[e, 'rank'])}. Kunci jawaban: jujur (look-alike C1).")
    st.error("**Batasan yang jujur:** angka ini hasil metode pada data **sintetis** dengan asumsi yang dilaporkan sebagai "
             "rentang (prevalensi, jam pemeriksaan, masa tagih). Satu seed. Perusahaan yang **tidak pernah mendaftar** "
             "sama sekali tidak terlihat oleh TEPAT. Tingkat kematangan: prototipe.")

# --------------------------------------------------------------------------- 4. fairness / random arm

with tab_fair:
    rb = load_results()["branches"][branch]
    arm = rb["rankings"].get(f"{rb.get('final_ranking', 'tepat_v2_1')}_with_random_arm", {})
    st.subheader("Apakah TEPAT memeriksa kelompok tertentu secara berlebihan?")
    st.markdown(f"**Jalur acak 8%:** {arm.get('random_arm_size', 0)} perusahaan dipilih acak berstrata (sektor × skala). "
                f"Tingkat pelanggaran di jalur acak: **{arm.get('random_arm_violation_rate', 0):.0%}**. Angka ini adalah "
                "perkiraan tak bias prevalensi sebenarnya. Jalur ini juga memberi data latih baru yang tidak bias seleksi.")
    fr = pd.DataFrame(rb.get("fairness", []))
    if len(fr):
        for grp, title in (("sector", "Sektor"), ("band_obs", "Skala usaha")):
            g = fr[fr.group == grp].copy().rename(columns={"value": "grp"})
            if grp == "sector":
                g["grp"] = g.grp.map(SEKTOR)
            m = g.melt(id_vars=["grp"], value_vars=["violation_rate", "pick_rate", "honest_flag_rate"], value_name="rate")
            m["variable"] = m.variable.map({"violation_rate": "Tingkat pelanggaran (sebenarnya)",
                                            "pick_rate": "Tingkat dipilih TEPAT",
                                            "honest_flag_rate": "Perusahaan jujur ikut terpilih"})
            st.altair_chart(alt.Chart(m).mark_bar().encode(
                x=alt.X("grp:N", title=title), xOffset="variable:N", y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title=None),
                color=alt.Color("variable:N", legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["grp", "variable", alt.Tooltip("rate:Q", format=".1%")]).properties(height=260, title=title),
                width="stretch")
    st.caption("Skala usaha mikro & kecil: upah di bawah UMK bukan pelanggaran (Perpres 59/2024 Ps. 32(4), E05c), jadi aturan itu "
               "hanya berlaku untuk usaha menengah & besar.")

# --------------------------------------------------------------------------- 5. worker self-check (D-011: B + C)

with tab_worker:
    st.subheader("Cek Kepesertaan Saya")
    st.caption("Untuk pekerja: apakah pemberi kerja sudah mendaftarkan dan membayar iuran BPJS Kesehatan Anda? "
               "Di dunia nyata fitur ini berada di Mobile JKN; di sini memakai data sintetis.")
    ks, tk, bill, mem = obs["ks_membership"], obs["tk_exchange"], obs["billing"], obs["ks_members"]

    @st.cache_data
    def demo_niks(branch: str) -> dict:
        tk7 = tk[(tk.quarter == 7) & (tk.program == "regular")]
        active_ks = ks[ks.ppu_end.isna() | (ks.ppu_end > D)]
        no_ks = tk7[~tk7.nik.isin(ks.nik) & ~tk7.nik.isin(mem.nik)]
        dep = tk7[tk7.nik.isin(mem.loc[mem.segment == "DEPENDENT", "nik"]) & ~tk7.nik.isin(ks.nik)]
        last = bill[(bill.agency == "KS") & (bill.month == R.HORIZON_MONTHS - 1)]
        unpaid = set(last.loc[last.paid_rp < last.billed_rp * 0.995, "employer_id"])
        susp = active_ks[active_ks.employer_id.isin(unpaid)]
        ok_emp = set(last.loc[last.paid_rp >= last.billed_rp, "employer_id"])
        ok = active_ks[active_ks.employer_id.isin(ok_emp)]
        out = {}
        if len(no_ks):
            out["Tidak didaftarkan pemberi kerja"] = no_ks.nik.iloc[0]
        if len(susp):
            out["Iuran tidak dibayar pemberi kerja"] = susp.nik.iloc[0]
        if len(dep):
            out["Hanya tercatat sebagai tanggungan"] = dep.nik.iloc[0]
        if len(ok):
            out["Semua beres"] = ok.nik.iloc[len(ok) // 2]
        return out

    demos = demo_niks(branch)
    pick = st.radio("Contoh NIK sintetis", list(demos), horizontal=True)
    nik = st.text_input("NIK (16 digit)", value=demos.get(pick, ""), max_chars=16)
    if nik:
        my_ks = ks[ks.nik == nik].sort_values("ppu_start")
        my_tk = tk[(tk.nik == nik) & (tk.quarter == tk[tk.nik == nik].quarter.max())] if (tk.nik == nik).any() else tk.iloc[0:0]
        my_dep = mem[(mem.nik == nik) & (mem.segment == "DEPENDENT")]
        active = my_ks[my_ks.ppu_end.isna() | (my_ks.ppu_end > D)]
        alerts = []
        k1, k2, k3, k4 = st.columns(4)
        if len(active):
            s = active.iloc[-1]
            emp = s.employer_id
            last = bill[(bill.employer_id == emp) & (bill.agency == "KS")].sort_values("month").tail(3)
            unpaid = last[last.paid_rp < last.billed_rp * 0.995]
            k1.metric("Terdaftar BPJS Kesehatan", "Ya", help=f"Sejak {day_label(s.ppu_start)}")
            k2.metric("Upah yang dilaporkan", rp(s.wage_reported))
            k3.metric("Iuran 3 bulan terakhir", "Lunas" if unpaid.empty else f"{len(unpaid)} bulan kurang")
            k4.metric("Status kepesertaan", "Aktif" if unpaid.empty else "Nonaktif sementara")
            st.markdown(f"Pemberi kerja: **{reg.at[emp, 'name']}**. Iuran {rp(R.iuran_ks(s.wage_reported, s.wage_reported, 'micro'))}/bulan "
                        f"(4% pemberi kerja, 1% dipotong dari gaji Anda).")
            if not unpaid.empty:
                alerts.append((emp, "iuran_tidak_dibayar",
                               f"Iuran BPJS Kesehatan bulan {', '.join(unpaid.month.map(month_label))} belum dibayar penuh oleh "
                               f"**{reg.at[emp, 'name']}**. Kepesertaan Anda dapat dinonaktifkan. Selama nonaktif, biaya perawatan "
                               "Anda menjadi tanggung jawab pemberi kerja (Perpres 82/2018 Ps. 42 ayat 2)."))
            tkw = tk[(tk.nik == nik) & (tk.employer_id == emp)]
            if len(tkw) and tkw.wage_reported_tk.iloc[-1] > s.wage_reported * 1.02:
                st.caption("Upah yang dilaporkan ke BPJS Ketenagakerjaan lebih tinggi. Apakah upah di atas sesuai slip gaji Anda?")
        else:
            k1.metric("Terdaftar BPJS Kesehatan", "Tidak" if my_dep.empty else "Sebagai tanggungan")
            k2.metric("Upah yang dilaporkan", "-")
            k3.metric("Iuran", "-")
            k4.metric("Status kepesertaan", "Tidak aktif" if my_dep.empty else "Aktif (tanggungan)")
        for _, row in my_tk.iterrows():
            emp = row.employer_id
            if emp == "BPU" or (len(active) and emp in set(active.employer_id)):
                continue
            jenis = "terdaftar_tanggungan" if len(my_dep) else "tidak_didaftarkan"
            alerts.append((emp, jenis,
                           f"BPJS Ketenagakerjaan mencatat Anda bekerja di **{reg.at[emp, 'name']}**, tetapi pemberi kerja belum "
                           "mendaftarkan Anda ke BPJS Kesehatan" + (" (Anda hanya tercatat sebagai tanggungan pasangan)" if len(my_dep) else "")
                           + ". Pemberi kerja wajib mendaftarkan setiap pekerja (UU 24/2011 Ps. 15; Perpres 82/2018 Ps. 13 & 14)."))
        if not alerts and len(active):
            st.success("Semua beres: Anda terdaftar dan iuran 3 bulan terakhir sudah dibayar.")
        for i, (emp, jenis, text) in enumerate(alerts):
            st.error("⚠️ " + text)
            if st.button("📣 Laporkan ke BPJS Kesehatan", key=f"lapor_{nik}_{i}"):
                tiket = add_tip(nik, emp, jenis)
                st.success(f"Laporan diterima. Nomor tiket **{tiket}**. Identitas Anda dilindungi (NIK disimpan sebagai hash); "
                           "laporan ini tampil di halaman kasus perusahaan tersebut untuk petugas pemeriksa.")
        if len(my_ks) == 0 and len(my_tk) == 0 and len(my_dep) == 0:
            st.info("NIK tidak ditemukan di data sintetis.")
    st.caption("Dalam proposal: notifikasi proaktif lewat Mobile JKN dan alur triase laporan (slide), bukan bagian prototipe.")
