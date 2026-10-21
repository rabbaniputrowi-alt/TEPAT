"""Read-only synthetic fairness diagnostics, isolated from operational case actions."""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from tepat.fairness_eval import COUNTS, RATES, SEEDS, aggregate_groups

DIMENSIONS = {"kabkota": "Kab/kota fiktif", "sector": "Sektor", "band_obs": "Band ukuran teramati"}


def render(branch: str, directory: Path, benchmark: dict, sector_labels: dict[str, str],
           chart_config: dict | None = None) -> None:
    """Show committed full-population evaluations in both demo and local modes."""
    arm = benchmark["rankings"].get(f"{benchmark.get('final_ranking', 'tepat_v2_1')}_with_random_arm", {})
    with st.container(key="summary"):
        a, b, c = st.columns(3)
        a.metric("Anggaran jalur acak", "8%")
        a.caption("Bagian dari anggaran jam pemeriksaan, bukan kuota perusahaan.")
        b.metric("Perusahaan sampel acak", str(arm.get("random_arm_size", "Tidak tersedia")))
        rate = arm.get("random_arm_violation_rate")
        c.metric("Pelanggaran di sampel acak", f"{rate:.0%}" if rate is not None and arm.get("random_arm_size", 0) else "Tidak terdefinisi")
    st.info("Jalur acak berstrata menurut band ukuran saja, bukan sektor atau wilayah. Sampel di luar daftar berisiko tinggi "
            "membantu diagnostik seleksi; bukan angka prevalensi nasional. Angka jalur acak di atas berasal dari seed bawaan.")
    st.caption("Evaluasi ini hanya memakai data sintetis dan kunci jawaban simulasi. Nama kab/kota fiktif dan nilai UMK "
               "bukan wilayah atau ketetapan nyata. Selisih antar kelompok bukan bukti bias hilang atau keadilan dunia nyata, "
               "dan tidak menetapkan kuota pemilihan otomatis.")
    if branch == "A":
        st.warning("Cabang A untuk tuning dan pelatihan saja, bukan hasil uji tertahan. B adalah uji tertahan; C adalah uji bergeser.")
    else:
        st.caption("Cabang B adalah uji tertahan. Cabang C menggeser campuran sektor, look-alike dan varian. Pelatihan hanya pada A.")
    st.markdown("**Definisi penyebut**")
    st.caption("n_total: semua perusahaan dalam kelompok; n_selected: terpilih dalam anggaran jam; n_violators: pelanggar yang ditanam; "
               "n_honest: perusahaan tanpa pelanggaran yang ditanam; n_honest_selected: perusahaan jujur yang terpilih. "
               "pick_rate = n_selected / n_total; violation_rate = n_violators / n_total; "
               "honest_flag_rate = n_honest_selected / n_honest. Ini tingkat penandaan jujur, bukan proporsi jujur di antara pilihan.")
    st.caption("Penyebut nol berarti tidak terdefinisi, bukan 0%. small_n menandai n_total < 30 atau n_honest < 30 "
               "sebagai peringatan deskriptif, bukan ambang signifikansi. __missing__ berarti label teramati tidak tersedia.")
    dimension = st.selectbox("Kelompok evaluasi", list(DIMENSIONS), format_func=DIMENSIONS.get, key="fairness_dimension")
    default_path = directory / "runs" / f"seed_{SEEDS[0]}.json"
    st.subheader(f"Seed bawaan {SEEDS[0]}")
    st.caption("TEPAT v2.1 tanpa jalur acak, seluruh populasi terdaftar pada konfigurasi bawaan. "
               "Tabel ini berasal dari evaluasi tersimpan, bukan hasil tindak lanjut petugas atau pelatihan ulang dalam aplikasi.")
    if not default_path.is_file():
        st.warning("Evaluasi seed bawaan belum tersedia. Tidak ada nilai nol pengganti.")
        return
    payload = json.loads(default_path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload["groups"])
    default = rows[(rows.branch == branch) & (rows.group == dimension)].copy()
    if default.empty:
        st.warning("Kelompok pada seed bawaan tidak tersedia, bukan kelompok dengan tingkat 0%.")
        return
    default["label"] = default.value.map(sector_labels).fillna(default.value) if dimension == "sector" else default.value
    if default.small_n.any():
        st.warning("Kelompok small-n: baca jumlah perusahaan dan penyebut jujur sebelum membandingkan tingkat.")
    if default[list(RATES)].isna().any().any():
        st.warning("Ada tingkat tidak terdefinisi karena penyebut nol. Sel kosong bukan hasil terukur 0%.")
    st.dataframe(default[["group", "value", "label"] + COUNTS + list(RATES) + ["small_n", "warning"]],
                 hide_index=True, width="stretch", column_config={
                     "pick_rate": st.column_config.NumberColumn("Tingkat terpilih", format="percent"),
                     "violation_rate": st.column_config.NumberColumn("Pelanggaran ditanam", format="percent"),
                     "honest_flag_rate": st.column_config.NumberColumn("Jujur terpilih", format="percent")})
    plotted = default.melt(id_vars=["label"] + COUNTS, value_vars=list(RATES), var_name="metric", value_name="rate")
    plotted["metric"] = plotted.metric.map({"pick_rate": "Terpilih", "violation_rate": "Pelanggaran ditanam", "honest_flag_rate": "Jujur terpilih"})
    chart = alt.Chart(plotted.dropna(subset=["rate"])).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("label:N", title=None, axis=alt.Axis(labelAngle=-20, labelLimit=180)), xOffset="metric:N",
        y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("metric:N", title=None, legend=alt.Legend(orient="bottom"),
                        scale=alt.Scale(domain=["Terpilih", "Pelanggaran ditanam", "Jujur terpilih"],
                                        range=["#087f82", "#98aeb5", "#d48b4a"])),
        tooltip=["label", "metric", alt.Tooltip("rate:Q", format=".1%")]
                + [alt.Tooltip(f"{col}:Q", format="d") for col in COUNTS]).properties(height=240)
    if chart_config:
        chart = chart.configure(**chart_config)
    st.altair_chart(chart, theme=None, width="stretch")
    st.caption("Grafik hanya menggambar tingkat terdefinisi. Penyebut dan kelompok tanpa nilai tetap ada di tabel.")
    st.caption("Band ukuran dihitung dari pekerja teramati, bukan skala usaha legal. Untuk usaha mikro dan kecil, "
               "upah di bawah UMK bukan pelanggaran (Perpres 59/2024 Ps. 32(4), E05c). Label pelanggaran hanya untuk evaluasi sintetis.")

    st.subheader("Variasi antar lima seed")
    st.caption("Seed " + ", ".join(map(str, SEEDS)) + ". Setiap seed membangkitkan ulang A/B/C. "
               "Ini variasi tingkat kelompok, bukan kestabilan peringkat perusahaan; ID yang sama tidak berarti entitas yang sama. "
               "Rerata memberi bobot sama pada seed dengan tingkat terdefinisi. Min s.d. maks bukan interval kepercayaan.")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for seed in SEEDS
                if (path := directory / "runs" / f"seed_{seed}.json").is_file()]
    all_rows = pd.DataFrame([row for item in payloads for row in item["groups"]])
    summary = aggregate_groups(all_rows)
    summary = summary[(summary.branch == branch) & (summary.group == dimension)].copy()
    if summary.empty:
        st.warning("Belum ada evaluasi lintas seed untuk kelompok ini. Tidak ada tingkat 0% pengganti.")
        return
    if summary.n_seeds_present.lt(len(SEEDS)).any():
        st.warning("Cakupan seed belum lengkap. Seed atau kelompok yang hilang tidak dihitung sebagai nol; lihat seeds_missing.")
    if summary.small_n_seeds.gt(0).any():
        st.warning("Ada kelompok small-n pada sebagian seed. Periksa small_n_seeds dan rentang jumlah pada tabel penyebut.")
    if summary[[f"{rate}_undefined_seeds" for rate in RATES]].gt(0).any().any():
        st.warning("Ada tingkat lintas seed tidak terdefinisi karena penyebut nol. Rerata dan rentang hanya memakai seed "
                   "terdefinisi; lihat n_seeds dan undefined_seeds untuk setiap tingkat.")
    summary["label"] = summary.value.map(sector_labels).fillna(summary.value) if dimension == "sector" else summary.value
    coverage = ["n_seeds_present", "n_seeds_expected", "n_nonempty_seeds", "seeds_missing", "small_n_seeds"]
    rate_columns = [f"{rate}_{stat}" for rate in RATES for stat in ["mean", "min", "max", "n_seeds", "undefined_seeds"]]
    st.dataframe(summary[["group", "value", "label"] + coverage + rate_columns], hide_index=True, width="stretch",
                 column_config={f"{rate}_{stat}": st.column_config.NumberColumn(format="percent")
                                for rate in RATES for stat in ["mean", "min", "max"]})
    st.caption("Setiap tingkat memiliki kolom n_seeds sendiri: jumlah seed dengan penyebut positif. undefined_seeds hanya "
               "menghitung baris hadir dengan penyebut nol; seeds_missing menyebut baris yang tidak hadir. Tidak ada imputasi nol.")
    st.markdown("**Jumlah dan penyebut lintas seed**")
    st.caption("Kolom _sum adalah jumlah pengamatan pada populasi seed yang tersedia, bukan perusahaan unik. "
               "Kolom _min dan _max adalah rentang jumlah per seed. Penyebut tidak dirata-ratakan. "
               "Semua angka diagnostik, bukan usulan kuota seleksi.")
    count_columns = [f"{count}_{stat}" for count in COUNTS for stat in ["sum", "min", "max"]]
    st.dataframe(summary[["group", "value", "label"] + coverage + count_columns], hide_index=True, width="stretch")
