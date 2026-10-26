"""Daftar prioritas, laid out after Claude Design turn 3a (merged workspace).

Numbers on this view come from three places, and the copy says which:
- KPI cards, list cards and the case panel: the model's estimates (ranked.parquet).
- The recovery curve: the synthetic answer key (truth/employers_truth.parquet), for both the TEPAT order and the
  rules-only order, so the two lines are valued the same way.
- The 2,3x and 95% badges: the saved benchmark at its fixed 10% budget (results.json), not the slider budget.

An 8% random arm is held back from the slider hours: only 92% of them go to the ranked queue. The language switch
covers this view's labels only; reasons, actions, firm and region names stay in Bahasa.
"""
from __future__ import annotations

from html import escape
from typing import Callable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

HOLDBACK = 0.08
PRESETS = (0.05, 0.10, 0.15, 0.25)
CARDS = 9
TIER_EN = {"Tinggi": "High", "Sedang": "Medium", "Rendah": "Low"}

T = {
    "ID": {
        "title": "Daftar prioritas",
        "sub": "Tentukan kapasitas pemeriksaan. Dahulukan potensi pemulihan per jam tertinggi.",
        "branch": "Cabang data sintetis", "route": "Rute tindak lanjut", "lang": "Bahasa",
        "insp": "Pemeriksaan", "inst": "Cicilan", "synth": "Data sintetis, satu seed",
        "branches": {"A": "Cabang A (tuning)", "B": "Cabang B (uji tertahan)", "C": "Cabang C (uji bergeser)"},
        "f_rec": "Perkiraan pemulihan", "f_firms": "Perusahaan dalam anggaran", "f_rph": "Perkiraan Rp per jam",
        "f_inst": "Antrean cicilan", "of_queue": "dari {n} perusahaan di antrean pemeriksaan",
        "inst_note": "Diarahkan ke program cicilan", "vs_rules": "dibanding aturan saja",
        "prec": "presisi pemeriksaan", "bench": "benchmark, anggaran 10%",
        "curve_title": "Kurva pemulihan kumulatif",
        "curve_note": "Rp dipulihkan menurut jam pemeriksaan, dinilai dengan kunci jawaban sintetis untuk kedua urutan.",
        "rules": "Aturan saja", "x_title": "Jam pemeriksaan", "y_title": "Rp jt dipulihkan (kunci jawaban)",
        "cut": "Kunci jawaban: {rp} pada {h} jam efektif, {n} perusahaan",
        "capacity": "Kapasitas pemeriksaan", "hrs": "jam", "slider": "Anggaran jam pemeriksaan",
        "slider_help": "Anggaran seluruh cabang, bukan hanya baris setelah filter.",
        "arm": "Jalur acak 8% ({h} jam) disisihkan untuk pemeriksaan acak",
        "start": "Mulai tindak lanjut",
        "fresh": "{a} dari {b} perusahaan dalam anggaran belum punya tindak lanjut. Memulai tindak lanjut hanya membuat "
                 "kasus berstatus Baru; tidak ada surat yang dikirim.",
        "started": "{n} kasus baru dibuat. Lihat halaman Tindak lanjut.",
        "list_title": "Perusahaan untuk ditinjau", "tier_label": "Tingkat",
        "tiers": {"Semua": "Semua", "Tinggi": "Tinggi", "Sedang": "Sedang"},
        "pick": "Pilih {name}", "kab_filtered": "Filter Kab/Kota aktif: {a} dari {b} wilayah.",
        "empty": "Tidak ada perusahaan sesuai filter. Pilih tingkat atau wilayah lain.",
        "rank": "Peringkat", "p_short": "P(pelanggaran)", "hrs_long": "jam periksa", "workers": "pekerja di BPJS TK",
        "reasons": "Alasan peringkat", "basis": "Dasar", "no_reason": "Tidak ada sinyal pelanggaran. Jalur hijau.",
        "open": "Buka kasus", "in_budget": "dalam anggaran", "out_budget": "di luar anggaran",
        "table": "Tabel lengkap", "kab": "Kab/Kota", "shown": "{n} perusahaan ditampilkan. Ringkasan anggaran mencakup "
                 "seluruh cabang; filter hanya mengubah daftar.",
        "cols": {"Peringkat": "#", "Perusahaan": "Perusahaan", "P(pelanggaran)": "P(pelanggaran)", "Rp/jam": "Rp/jam",
                 "Tingkat": "Tingkat", "Anggaran": "Anggaran", "Kab/Kota": "Kab/Kota", "Sektor": "Sektor",
                 "Skala": "Skala", "Perkiraan Rp": "Perkiraan Rp", "Rentang": "Rentang",
                 "Alasan utama": "Alasan utama", "Tindakan": "Tindakan"},
        "selected": "Terpilih", "outside": "Di luar anggaran", "range": "{a} s.d. {b}",
        "foot": "Petugas tetap mengambil keputusan. Tidak ada keputusan otomatis semata (UU PDP Ps. 10). Badge 2,3x dan "
                "presisi berasal dari benchmark tersimpan dengan anggaran 10%, bukan dari anggaran slider.",
    },
    "EN": {
        "title": "Priority queue",
        "sub": "Set inspection capacity. Put the highest recovery per hour first.",
        "branch": "Synthetic data branch", "route": "Follow-up route", "lang": "Language",
        "insp": "Inspection", "inst": "Installments", "synth": "Synthetic data, one seed",
        "branches": {"A": "Branch A (tuning)", "B": "Branch B (held out)", "C": "Branch C (shifted)"},
        "f_rec": "Estimated recovery", "f_firms": "Employers in budget", "f_rph": "Estimated Rp per hour",
        "f_inst": "Installment queue", "of_queue": "of {n} employers in the inspection queue",
        "inst_note": "Routed to installment plans", "vs_rules": "versus rules only",
        "prec": "inspection precision", "bench": "benchmark, 10% budget",
        "curve_title": "Cumulative recovery curve",
        "curve_note": "Rp recovered by inspection hours, valued with the synthetic answer key for both orders.",
        "rules": "Rules only", "x_title": "Inspection hours", "y_title": "Rp million recovered (answer key)",
        "cut": "Answer key: {rp} at {h} effective hrs, {n} employers",
        "capacity": "Inspection capacity", "hrs": "hrs", "slider": "Inspection hour budget",
        "slider_help": "Budget for the whole branch, not only the filtered rows.",
        "arm": "Random arm 8% ({h} hrs) held back for random inspections",
        "start": "Start follow-up",
        "fresh": "{a} of {b} employers in budget have no follow-up yet. Starting follow-up only creates cases with "
                 "status Baru; no letter is sent.",
        "started": "{n} new cases created. See the Tindak lanjut page.",
        "list_title": "Employers to review", "tier_label": "Tier",
        "tiers": {"Semua": "All", "Tinggi": "High", "Sedang": "Medium"},
        "pick": "Select {name}", "kab_filtered": "Regency filter on: {a} of {b} regions.",
        "empty": "No employers match the filter. Choose another tier or region.",
        "rank": "Rank", "p_short": "P(violation)", "hrs_long": "inspection hrs", "workers": "workers on BPJS TK",
        "reasons": "Why it ranks here", "basis": "Basis", "no_reason": "No violation signal. Green lane.",
        "open": "Open case", "in_budget": "in budget", "out_budget": "outside budget",
        "table": "Full table", "kab": "Regency/City", "shown": "{n} employers shown. The budget summary covers the whole "
                 "branch; filters only change the list.",
        "cols": {"Peringkat": "#", "Perusahaan": "Employer", "P(pelanggaran)": "P(violation)", "Rp/jam": "Rp/hr",
                 "Tingkat": "Tier", "Anggaran": "Budget", "Kab/Kota": "Regency/City", "Sektor": "Sector",
                 "Skala": "Size", "Perkiraan Rp": "Estimated Rp", "Rentang": "Range",
                 "Alasan utama": "Main reason", "Tindakan": "Action"},
        "selected": "Selected", "outside": "Outside budget", "range": "{a} to {b}",
        "foot": "Inspectors still decide. No solely automated decision (PDP Law Art. 10). The 2.3x and precision badges "
                "come from the saved benchmark at a 10% budget, not from the slider budget.",
    },
}


# ------------------------------------------------------------------ pure helpers (tested)

def num(x: float, lang: str, d: int = 0) -> str:
    """1.500,5 in Bahasa, 1,500.5 in English."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".") if lang == "ID" else s


def money(x: float, lang: str) -> str:
    """Rupiah in millions: Rp7.436,1 jt or Rp7,436.1M."""
    return "Rp" + num(x / 1e6, lang, 1) + (" jt" if lang == "ID" else "M")


def ratio(x: float, lang: str) -> str:
    return num(x, lang, 1) + "x"


def effective_hours(budget: float, holdback: float = HOLDBACK) -> float:
    return budget * (1 - holdback)


def select_in_budget(hours: pd.Series, budget: float, holdback: float = HOLDBACK) -> pd.Index:
    """Employers (in queue order) that fit the budget after the random-arm holdback."""
    return hours.index[hours.cumsum() <= effective_hours(budget, holdback)]


def preset_hours(total_hours: float) -> list[int]:
    return [int(round(total_hours * f)) for f in PRESETS]


def cumulative(order: pd.Index, hours: pd.Series, value: pd.Series) -> pd.DataFrame:
    """Cumulative hours and value along an inspection order, starting at (0, 0)."""
    h = np.concatenate([[0.0], hours.loc[order].to_numpy(float).cumsum()])
    v = np.concatenate([[0.0], value.loc[order].to_numpy(float).cumsum()])
    return pd.DataFrame({"jam": h, "rp": v})


def recovery_curves(tepat_order: pd.Index, rules_order: pd.Index, hours: pd.Series, value: pd.Series,
                    xmax: float, labels: tuple[str, str] = ("TEPAT v2.1", "Aturan saja")) -> pd.DataFrame:
    """Both orders valued with the same per-employer value, cut at xmax hours (one point past it is clipped)."""
    parts = []
    for order, label in zip((tepat_order, rules_order), labels):
        c = cumulative(order, hours, value)
        parts.append(c[c.jam <= xmax].assign(metode=label))
    return pd.concat(parts, ignore_index=True)


def value_at(order: pd.Index, hours: pd.Series, value: pd.Series, limit: float) -> tuple[float, int]:
    """Value and count of the employers that fit within `limit` hours along the order."""
    fit = order[hours.loc[order].cumsum().to_numpy() <= limit]
    return float(value.loc[fit].sum()), len(fit)


# ------------------------------------------------------------------ rendering

def _kpi(label: str, value: str, foot: str, dark: bool = False) -> str:
    cls = "kpi-card kpi-dark" if dark else "kpi-card"
    return (f'<div class="{cls}"><div class="kpi-label">{escape(label)}</div>'
            f'<div class="kpi-value">{escape(value)}</div><div class="kpi-foot">{foot}</div></div>')


def _badge(text: str, dark: bool = False) -> str:
    return f'<span class="{"kpi-badge kpi-badge-dark" if dark else "kpi-badge"}">{escape(text)}</span>'


def _select(branch: str, eid: str) -> None:
    st.session_state["queue_sel"] = (branch, eid)


def _set_budget(v: int) -> None:
    st.session_state["budget"] = v


def render(*, lang: str, branch: str, route: str, L: pd.DataFrame, F: pd.DataFrame, truth: pd.DataFrame, rb: dict,
           store, reasons: Callable[[pd.Series], list], sektor: dict, chart_config: dict,
           open_case: Callable[[str], None]) -> None:
    t = T[lang]
    insp = L[L.queue == "inspeksi"]
    total_h = float(F.hours.sum())
    lo, hi = 50, int(total_h * 0.3)
    if "budget" not in st.session_state:
        st.session_state["budget"] = int(total_h * 0.10)
    st.session_state["budget"] = int(min(max(st.session_state["budget"], lo), hi))
    budget = st.session_state["budget"]
    eff = effective_hours(budget)
    in_budget = set(select_in_budget(insp.hours, budget))
    sel = L.loc[[e for e in insp.index if e in in_budget]]
    final = rb.get("final_ranking", "tepat_v2_1")
    best, base = rb["rankings"][final], rb["rankings"]["rules_only"]

    # KPI cards ------------------------------------------------------------------------------
    rph = sel.expected_rp.sum() / max(1.0, sel.hours.sum())
    cards = [
        _kpi(t["f_rec"], money(sel.expected_rp.sum(), lang),
             _badge(ratio(best["rp_per_hour"] / max(1.0, base["rp_per_hour"]), lang), dark=True)
             + f' {escape(t["vs_rules"])} <span class="kpi-src">({escape(t["bench"])})</span>', dark=True),
        _kpi(t["f_firms"], num(len(sel), lang), escape(t["of_queue"].format(n=num(len(insp), lang)))),
        _kpi(t["f_rph"], money(rph, lang),
             _badge(f"{best['precision']:.0%}") + f' {escape(t["prec"])} <span class="kpi-src">({escape(t["bench"])})</span>'),
        _kpi(t["f_inst"], num(int((L.queue == "cicilan").sum()), lang), escape(t["inst_note"])),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

    # Filters shared by the cards and the full table -----------------------------------------
    all_kab = sorted(L.kabkota.unique())
    kab = [k for k in st.session_state.get("districts", all_kab) if k in all_kab] or all_kab
    tier = st.session_state.get("queue_tier", "Semua")
    V = L[L.kabkota.isin(kab) & (L.queue == route)]
    if tier != "Semua":
        V = V[V.tier == tier]

    curve_col, list_col, case_col = st.columns([1.5, 1.15, 0.95], gap="medium")

    # 1. Curve and capacity ---------------------------------------------------------------------
    with curve_col, st.container(border=True, key="curve_card"):
        value = truth.recoverable_rp_true.reindex(F.index).fillna(0.0)
        rules_order = _rules_order(F)
        curves = recovery_curves(insp.index, rules_order, F.hours, value, hi, ("TEPAT v2.1", t["rules"]))
        cut_rp, cut_n = value_at(insp.index, F.hours, value, eff)
        st.markdown(
            f'<div class="card-head"><div><div class="card-title">{escape(t["curve_title"])}</div>'
            f'<div class="card-note">{escape(t["curve_note"])}</div></div>'
            f'<div class="legend"><span><i class="sw sw-tepat"></i>TEPAT v2.1</span>'
            f'<span><i class="sw sw-rules"></i>{escape(t["rules"])}</span></div></div>'
            f'<div class="cut-chip">{escape(t["cut"].format(rp=money(cut_rp, lang), h=num(eff, lang), n=num(cut_n, lang)))}</div>',
            unsafe_allow_html=True)
        st.altair_chart(_curve_chart(curves, eff, cut_rp, lang, t, chart_config), theme=None, width="stretch")
        with st.container(key="capacity"):
            presets = preset_hours(total_h)
            st.markdown(f'<div class="cap-label">{escape(t["capacity"])}</div>'
                        f'<div class="cap-value">{escape(num(budget, lang))} {escape(t["hrs"])}</div>', unsafe_allow_html=True)
            for col, v in zip(st.columns(len(presets), gap="small"), presets):
                with col, st.container(key=f"preset_{'on' if v == budget else 'off'}_{v}"):
                    st.button(num(v, lang), key=f"preset_{v}", on_click=_set_budget, args=(v,), width="stretch")
            st.slider(t["slider"], lo, hi, step=25, key="budget", help=t["slider_help"], label_visibility="collapsed")
            fresh = [e for e in insp.index if e in in_budget and store.get(branch, e) is None]
            st.caption(t["arm"].format(h=num(budget * HOLDBACK, lang)))
            if st.button(t["start"], key="bulk_follow_up", disabled=not fresh, type="primary", width="stretch"):
                created, _ = store.bulk_open(branch, L.loc[fresh, ["est_rp", "p_violation", "action"]])
                st.session_state["bulk_flash"] = t["started"].format(n=created)
                st.rerun()
            st.caption(t["fresh"].format(a=len(fresh), b=len(in_budget)))
            if msg := st.session_state.pop("bulk_flash", None):
                st.success(msg)

    # 2. Card list ------------------------------------------------------------------------------
    with list_col, st.container(border=True, key="review_card"):
        st.markdown(f'<div class="card-title">{escape(t["list_title"])}</div>', unsafe_allow_html=True)
        st.radio(t["tier_label"], ["Semua", "Tinggi", "Sedang"], key="queue_tier", horizontal=True,
                 format_func=t["tiers"].get, label_visibility="collapsed")
        if len(kab) < len(all_kab):
            st.caption(t["kab_filtered"].format(a=len(kab), b=len(all_kab)))
        picked = st.session_state.get("queue_sel")
        eid = picked[1] if picked and picked[0] == branch and picked[1] in V.index else (V.index[0] if len(V) else None)
        if V.empty:
            st.info(t["empty"])
        for e, r in V.head(CARDS).iterrows():
            inb = e in in_budget
            with st.container(key=f"qcard_{'sel' if e == eid else 'row'}_{int(r['rank'])}"):
                st.markdown(
                    f'<div class="qcard"><span class="qrank {"qrank-in" if inb else ""}" '
                    f'title="{escape(t["in_budget"] if inb else t["out_budget"])}">{int(r["rank"])}</span>'
                    f'<span class="qname">{escape(r["name"])}</span>'
                    f'<span class="qrph">{escape(money(r.expected_rp / r.hours, lang))}</span>'
                    f'<span class="qbar"><span class="qtrack"><span class="qfill" style="width:{r.p_violation:.0%}"></span></span>'
                    f'<span class="qpct">{r.p_violation:.0%}</span></span>'
                    f'<span class="qkab">{escape(r.kabkota)}</span></div>', unsafe_allow_html=True)
                st.button(t["pick"].format(name=r["name"]), key=f"qpick_{e}", on_click=_select, args=(branch, e))

    # 3. Case panel -----------------------------------------------------------------------------
    with case_col, st.container(border=True, key="case_card"):
        if eid is None:
            st.info(t["empty"])
        else:
            r, f_ = L.loc[eid], F.loc[eid]
            tier_word = str(TIER_EN.get(r.tier, r.tier) if lang == "EN" else r.tier).lower()
            rs = reasons(f_)[:3]
            items = "".join(
                f'<div class="reason"><span class="rnum">{i}</span><span class="rtext">{escape(text)}</span>'
                f'<span></span><span class="rbasis">{escape(t["basis"])}: {escape(basis)}</span></div>'
                for i, (_, text, basis) in enumerate(rs, start=1)) or f'<div class="card-note">{escape(t["no_reason"])}</div>'
            st.markdown(
                f'<div class="case-preview-name">{escape(r["name"])}</div>'
                f'<div class="case-preview-detail">{escape(t["rank"])} {int(r["rank"])}, {escape(tier_word)}, '
                f'{escape(r.kabkota)}</div>'
                f'<div class="case-stats"><div><b>{r.p_violation:.0%}</b><span>{escape(t["p_short"])}</span></div>'
                f'<div><b>{int(f_.hours)}</b><span>{escape(t["hrs_long"])}</span></div>'
                f'<div><b>{escape(num(int(f_.tk_n), lang))}</b><span>{escape(t["workers"])}</span></div></div>'
                f'<div class="reasons-title">{escape(t["reasons"])}</div>{items}', unsafe_allow_html=True)
            st.button(t["open"], key="open_case", type="primary", width="stretch", on_click=open_case, args=(eid,))

    # Full table ----------------------------------------------------------------------------------
    with st.expander(t["table"], expanded=False):
        st.multiselect(t["kab"], all_kab, default=all_kab, key="districts")
        st.caption(t["shown"].format(n=num(len(V), lang)))
        c = t["cols"]
        tier_txt = V.tier.map(TIER_EN) if lang == "EN" else V.tier
        show = pd.DataFrame({
            c["Peringkat"]: V["rank"],
            c["Anggaran"]: np.where(V.index.isin(list(in_budget)), t["selected"], t["outside"]),
            c["Perusahaan"]: V["name"], c["Sektor"]: V.sector.map(sektor), c["Kab/Kota"]: V.kabkota,
            c["Skala"]: V.band_obs, c["Tingkat"]: tier_txt, c["P(pelanggaran)"]: V.p_violation.round(2),
            c["Perkiraan Rp"]: V.est_rp.map(lambda x: money(x, lang)),
            c["Rentang"]: [t["range"].format(a=money(a, lang), b=money(b, lang)) for a, b in zip(V.rp_low, V.rp_high)],
            c["Rp/jam"]: (V.expected_rp / V.hours).map(lambda x: money(x, lang)),
            c["Alasan utama"]: V.top_reasons.str.split(" \\| ").str[0], c["Tindakan"]: V.action}, index=V.index)
        order = [c[k] for k in ("Peringkat", "Perusahaan", "P(pelanggaran)", "Rp/jam", "Tingkat", "Anggaran", "Kab/Kota",
                                "Sektor", "Skala", "Perkiraan Rp", "Rentang", "Alasan utama", "Tindakan")]
        st.dataframe(show, height=420, width="stretch", hide_index=True, column_order=order,
                     column_config={c["Peringkat"]: st.column_config.NumberColumn(c["Peringkat"], width="small"),
                                    c["Perusahaan"]: st.column_config.TextColumn(width="medium"),
                                    c["P(pelanggaran)"]: st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    st.caption(t["foot"])


def _rules_order(F: pd.DataFrame) -> pd.Index:
    from tepat import scorer
    return scorer.rules_only(F).index


def _curve_chart(curves: pd.DataFrame, eff: float, cut_rp: float, lang: str, t: dict, config: dict) -> alt.Chart:
    data = curves.assign(rp_jt=curves.rp / 1e6)
    sep = "." if lang == "ID" else ","
    fmt = f"replace(format(datum.value, ',.0f'), regexp(',', 'g'), '{sep}')"
    color = alt.Color("metode:N", scale=alt.Scale(domain=["TEPAT v2.1", t["rules"]], range=["#087f82", "#98aeb5"]), legend=None)
    x = alt.X("jam:Q", title=t["x_title"], axis=alt.Axis(labelExpr=fmt, tickCount=5, grid=False),
              scale=alt.Scale(domain=[0, float(data.jam.max())], nice=False))
    y = alt.Y("rp_jt:Q", title=t["y_title"], axis=alt.Axis(labelExpr=fmt, tickCount=4, gridDash=[3, 5]))
    lines = alt.Chart(data).mark_line(strokeWidth=3, strokeCap="round").encode(
        x=x, y=y, color=color, strokeWidth=alt.condition(alt.datum.metode == "TEPAT v2.1", alt.value(3), alt.value(2)))
    area = alt.Chart(data[(data.metode == "TEPAT v2.1") & (data.jam <= eff)]).mark_area(
        line=False, color=alt.Gradient(gradient="linear", x1=1, x2=1, y1=0, y2=1,
                                       stops=[alt.GradientStop(color="rgba(8,127,130,0.32)", offset=0),
                                              alt.GradientStop(color="rgba(8,127,130,0.02)", offset=1)])).encode(x="jam:Q", y="rp_jt:Q")
    marks = pd.DataFrame({"jam": [eff], "rp_jt": [cut_rp / 1e6]})
    rule = alt.Chart(marks).mark_rule(color="#123b40", strokeDash=[4, 4], strokeWidth=1.5).encode(x="jam:Q")
    dot = alt.Chart(marks).mark_point(filled=True, fill="#ffffff", stroke="#123b40", strokeWidth=3, size=120).encode(
        x="jam:Q", y="rp_jt:Q")
    return (area + lines + rule + dot).properties(height=245).configure(**{**config, "background": "transparent"}).configure_view(stroke=None)
