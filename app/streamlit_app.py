"""TEPAT demo app (synthetic data only).

    uv run streamlit run app/streamlit_app.py

Inspector views read observed/ + the scorer's ranked output. Only the clearly labelled
"kunci jawaban" expanders, the benchmark tab and the second list's test table read truth/ or
benchmark results (to show how the method was scored). The Benchmark tab also reads the committed
5-seed sweep (results/sweep/), and "Usaha belum terdaftar" reads the registration pass
(`tepat unreg`: results/unreg_ranked_<branch>.parquet and results/unregistered_*.csv).

Data source per branch: the full local run (data/<branch>/ + results/*.parquet, gitignored) when it
exists, otherwise the committed slice in demo/ (fresh clone, Streamlit Community Cloud). Set
TEPAT_DEMO=1 to force the slice. See scripts/make_demo_slice.py for what the slice keeps.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if importlib.util.find_spec("tepat") is None:  # Community Cloud installs requirements only, not the package
    sys.path.insert(0, str(ROOT / "src"))

from tepat import cases as K  # noqa: E402
from tepat import reports as P  # noqa: E402
from tepat import rules as R  # noqa: E402
from tepat import scorer  # noqa: E402
if str(ROOT / "app") not in sys.path:  # drafting_panel.py sits next to this script
    sys.path.insert(0, str(ROOT / "app"))
import case_panel  # noqa: E402
import drafting_panel  # noqa: E402
import fairness_panel  # noqa: E402
import queue_panel  # noqa: E402
import reports_panel  # noqa: E402
import unregistered_panel  # noqa: E402
import workspace_ui  # noqa: E402
from tepat import drafting  # noqa: E402

DATA = Path(os.environ.get("TEPAT_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("TEPAT_RESULTS", ROOT / "results"))
DEMO = ROOT / "demo"
TIPS = RESULTS / "tips.csv"
CASES_FILE = RESULTS / "cases.sqlite"
SUMBER_KASUS = {"contoh": "Contoh demo", "petugas": "Petugas", "massal": "Mulai massal", "laporan": "Laporan pekerja"}
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
SEKTOR = {"C": "Manufaktur", "F": "Konstruksi", "G": "Perdagangan", "H": "Logistik", "I": "Akomodasi & Makan",
          "J": "Informasi & Kom.", "K": "Keuangan", "Q": "Kesehatan", "A": "Pertanian"}
RANK_LABEL = {"random": "Acak", "biggest_first": "Terbesar dulu", "rules_only": "Aturan saja",
              "tepat_v1": "TEPAT v1", "tepat_v2": "TEPAT v2", "tepat_v2_1": "TEPAT v2.1",
              "tepat_v2_1_with_random_arm": "TEPAT v2.1 + jalur acak 8%"}
UNREG_LABEL = {"tepat_registration": "TEPAT daftar kedua", "rules_only": "Aturan hukum (TK dan OSS)",
               "oss_only": "Pencocokan OSS saja", "random": "Acak"}
SKALA = {"micro": "Mikro", "small": "Kecil", "medium": "Menengah", "large": "Besar"}
DIAL = {"default": "Bawaan", "prevalence": "Prevalensi pelanggar", "lookback_months": "Masa tagih (bulan)",
        "budget_frac": "Anggaran jam", "inspection_hours": "Jam pemeriksaan per skala"}
NOT_OBLIGED = "Tidak wajib"

st.set_page_config(page_title="TEPAT Pemeriksaan Kepatuhan", page_icon=":material/center_focus_strong:", layout="wide")


# --------------------------------------------------------------------------- helpers

def rp(x: float) -> str:
    return "Rp" + f"{x:,.0f}".replace(",", ".")


def ribu(n: int) -> str:
    """Indonesian thousands separator: 1.500, not 1,500."""
    return f"{n:,}".replace(",", ".")


def jt(x: float) -> str:
    return "Rp" + f"{x / 1e6:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".") + " jt"


def kali(x: float) -> str:
    """Ratio in the Indonesian form used on the slides: 2,3x."""
    return f"{x:.1f}".replace(".", ",") + "x"


def angka(x: float, d: int = 1) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct_range(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.0%} ({lo * 100:.0f} s.d. {hi:.0%})"


def month_label(m: int) -> str:
    return f"{BULAN[(9 + m) % 12]} {2024 + (9 + m) // 12}"


def day_label(d: float | None) -> str:
    if d is None or pd.isna(d):
        return "-"
    d = int(d)
    if d < 0:
        return "sebelum Okt 2024"
    return f"{d % 30 + 1} {month_label(d // 30)}"


def sources(branch: str) -> tuple[Path, Path, Path, bool]:
    """Branch folder, ranked and features files: the full local run if present, else the demo slice."""
    full = (DATA / branch, RESULTS / f"ranked_{branch}.parquet", RESULTS / f"features_{branch}.parquet")
    if os.environ.get("TEPAT_DEMO") != "1" and (full[0] / "observed").is_dir() and full[1].is_file() and full[2].is_file():
        return (*full, False)
    return DEMO / branch, DEMO / branch / "ranked.parquet", DEMO / branch / "features.parquet", True


@st.cache_data(show_spinner="Memuat data sintetis")
def load(branch_dir: str, ranked_path: str, features_path: str):
    obs = scorer.load_observed(branch_dir)
    ranked = pd.read_parquet(ranked_path)
    F = pd.read_parquet(features_path)
    reg = obs["employer_registry"].set_index("employer_id")
    return obs, ranked, F, reg


@st.cache_data
def load_truth(branch_dir: str) -> pd.DataFrame:
    return pd.read_parquet(Path(branch_dir) / "truth" / "employers_truth.parquet").set_index("employer_id")


@st.cache_data
def demo_worker_employers(branch: str) -> int:
    """How many employers the demo slice keeps worker rows for (0 if the manifest is missing)."""
    manifest = DEMO / "manifest.json"
    if not manifest.is_file():
        return 0
    return len(json.loads(manifest.read_text(encoding="utf-8"))["branches"][branch]["employers_with_worker_detail"])


@st.cache_data
def load_results() -> dict:
    # TEPAT_RESULTS may point at a writable scratch store (reports, cases) without a benchmark run in it;
    # the committed benchmark then stays the source for every badge and chart that cites it.
    path = RESULTS / "results.json"
    if not path.exists():
        path = ROOT / "results" / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def unreg_sources(branch: str, demo_mode: bool) -> tuple[Path, Path]:
    """Second list (registration pass): ranked candidates and their answer key. Same source rule as sources():
    the full local run when it has the files, otherwise the committed slice."""
    full = (RESULTS / f"unreg_ranked_{branch}.parquet", DATA / branch / "truth" / "unregistered_truth.parquet")
    if not demo_mode and full[0].is_file() and full[1].is_file():
        return full
    return DEMO / branch / "unreg_ranked.parquet", DEMO / branch / "truth" / "unregistered_truth.parquet"


@st.cache_data(show_spinner="Memuat daftar kedua")
def load_unreg(ranked_path: str, truth_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    U = pd.read_parquet(ranked_path)
    U.index = U.index.astype(str)
    T = pd.read_parquet(truth_path).assign(nib=lambda d: d.nib.astype(str)).set_index("nib")
    return U, T


def open_case(employer_id: str) -> None:
    """Preserve the selected queue case when opening the inspector view."""
    st.session_state["view"] = "Kasus perusahaan"
    st.session_state["case_employer"] = employer_id


def tanggal(iso: str | None) -> str:
    if not iso:
        return "-"
    d = datetime.fromisoformat(iso)
    return f"{d.day} {BULAN[d.month - 1]} {d.year}"


def waktu(iso: str) -> str:
    d = datetime.fromisoformat(iso)
    return f"{tanggal(iso)}, {d.hour:02d}.{d.minute:02d}"


@st.cache_resource
def persistent_store(path: str) -> K.CaseStore:
    return K.CaseStore.persistent(path)


def case_store(demo_mode: bool) -> K.CaseStore:
    """Local full run: one SQLite file that persists. Demo slice (public app): a private in-memory
    store per browser session, so visitors never see each other's edits. TEPAT_CASES overrides."""
    mode = os.environ.get("TEPAT_CASES") or ("sandbox" if demo_mode else "persistent")
    if mode == "persistent":
        return persistent_store(str(CASES_FILE))
    if "case_store" not in st.session_state:
        st.session_state["case_store"] = K.CaseStore.sandbox()
    return st.session_state["case_store"]


@st.cache_resource
def persistent_reports(path: str) -> P.ReportStore:
    return P.ReportStore(persistent_store(path))


def report_store(cases: K.CaseStore) -> P.ReportStore:
    if cases.mode == "persistent":
        reports = persistent_reports(str(cases.path))
        st.session_state["report_store"] = reports
    elif "report_store" not in st.session_state or st.session_state["report_store"].cases is not cases:
        st.session_state["report_store"] = P.ReportStore(cases)
    return st.session_state["report_store"]


def flash(message: str) -> None:
    st.session_state["fu_flash"] = message


def follow_up_panel(store: K.CaseStore, branch: str, eid: str, r: pd.Series) -> None:
    """Case page: start follow-up, change status, inspector and date, add notes, close with an outcome."""
    case = store.get(branch, eid)
    with st.container(border=True, key="follow_up"):
        st.subheader("Tindak lanjut")
        if msg := st.session_state.pop("fu_flash", None):
            st.success(msg)
        problem = st.session_state.pop("fu_error", None)
        if problem:
            st.error(problem)
        if case is None:
            st.caption("Belum ada tindak lanjut untuk perusahaan ini. Mulai bila petugas memutuskan untuk menindaklanjuti.")
            if st.button("Mulai tindak lanjut", key="fu_start", type="primary"):
                store.open_case(branch, eid, est_rp=r.est_rp, p_violation=r.p_violation, action=r.action)
                flash("Tindak lanjut dimulai dengan status Baru.")
                st.rerun()
            return
        cid = int(case["case_id"])
        # Widget keys carry the case and its history length, so after every saved change the form starts
        # again from the stored case instead of keeping the browser's previous inputs.
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
                note = st.text_area("Catatan", key=f"fu_note_{v}", height=80, placeholder="Misalnya hasil telepon ke bagian personalia")
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
                        except K.CaseError as e:
                            st.session_state["fu_error"] = str(e)
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
                        except K.CaseError as e:
                            st.session_state["fu_error"] = str(e)
                            st.rerun()
                        else:
                            flash("Kasus dibuka kembali dengan status Menunggu koreksi.")
                            st.rerun()
        with st.expander("Riwayat tindak lanjut"):
            ev = store.events(cid)
            st.dataframe(pd.DataFrame({"Waktu": ev["at"].map(waktu), "Oleh": ev.actor, "Jenis": ev.kind, "Keterangan": ev.detail}),
                         hide_index=True, width="stretch")


ASSETS = ROOT / "app" / "assets"
font_data = base64.b64encode((ASSETS / "public-sans-latin.woff2").read_bytes()).decode("ascii")
css = ((ASSETS / "workspace.css").read_text(encoding="utf-8") + "\n" +
       (ASSETS / "workspace-full.css").read_text(encoding="utf-8")).replace("__PUBLIC_SANS__", font_data)
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
VIEWS = ["Daftar prioritas", "Kasus perusahaan", "Tindak lanjut", "Laporan pekerja", "Usaha belum terdaftar", "Benchmark", "Keadilan & jalur acak",
         "Cek kepesertaan saya"]
DESCRIPTIONS = {
    "Daftar prioritas": "Tentukan kapasitas pemeriksaan. Dahulukan potensi pemulihan per jam tertinggi.",
    "Kasus perusahaan": "Telusuri sinyal, dasar hukum, dan dokumen sebelum mengambil keputusan.",
    "Tindak lanjut": "Pantau kasus yang sedang ditangani, jadwal berikutnya, dan hasil pemeriksaan.",
    "Laporan pekerja": "Tinjau dugaan pekerja, catat penelaahan, dan tautkan ke tindak lanjut bila diperlukan.",
    "Usaha belum terdaftar": "Usaha yang wajib mendaftar ke BPJS Kesehatan tetapi belum ada di registri. Daftar terpisah dari antrean utama.",
    "Benchmark": "Bandingkan metode dengan anggaran jam yang sama. Semua hasil berasal dari simulasi sintetis.",
    "Keadilan & jalur acak": "Pantau siapa yang terpilih dan siapa yang terlewat. Jangan hanya belajar dari kasus berisiko tinggi.",
    "Cek kepesertaan saya": "Apakah pemberi kerja sudah mendaftarkan Anda dan membayar iuran? Coba dengan NIK sintetis.",
}
CHART_CONFIG = {
    "background": "#ffffff", "view": {"stroke": None},
    "axis": {"labelFont": "Public Sans", "titleFont": "Public Sans", "labelFontSize": 11,
             "titleFontSize": 11, "labelColor": "#587078", "titleColor": "#365a60",
             "gridColor": "#e6eeee", "domain": False, "ticks": False, "labelPadding": 8,
             "titlePadding": 14, "labelLimit": 230},
    "legend": {"labelFont": "Public Sans", "titleFont": "Public Sans", "labelFontSize": 11,
               "labelColor": "#587078", "symbolType": "circle", "padding": 14},
}

# Persistent workspace shell -------------------------------------------------
NAV_ICON = {"Daftar prioritas": "format_list_numbered", "Kasus perusahaan": "work", "Tindak lanjut": "assignment_turned_in",
            "Laporan pekerja": "campaign", "Usaha belum terdaftar": "domain", "Benchmark": "bar_chart",
            "Keadilan & jalur acak": "balance", "Cek kepesertaan saya": "how_to_reg"}
BRANCH_LABEL = {"A": "A (tuning)", "B": "B (uji tertahan)", "C": "C (uji bergeser)"}
st.sidebar.markdown('<div class="sidebar-brand">TEPAT</div>', unsafe_allow_html=True)
view = st.sidebar.radio("Ruang kerja", VIEWS, key="view", label_visibility="collapsed",
                        format_func=lambda v: f":material/{NAV_ICON[v]}: {v}")
is_queue = view == "Daftar prioritas"
lang = st.session_state.get("queue_lang", "ID") if is_queue else "ID"
QT = queue_panel.T[lang]


def lang_key(base: str) -> str:
    """Relabelled options leave the browser widget with no visible selection while the server keeps the value,
    so each language gets its own key (the widget remounts) and the value carries over from the other one."""
    key, other = (base, f"{base}_en") if lang == "ID" else (f"{base}_en", base)
    if key not in st.session_state and other in st.session_state:
        st.session_state[key] = st.session_state[other]
    return key


head_l, head_r = st.container(key="page_head").columns([1.05, 1], gap="large", vertical_alignment="bottom")
with head_r:
    if is_queue:
        b_col, r_col, l_col = st.columns([1.15, 1.2, 0.72], gap="small", vertical_alignment="bottom")
    else:
        if view == "Kasus perusahaan":
            case_slot, b_col = st.columns([2.3, 1.2], gap="small", vertical_alignment="bottom")
        else:
            _, b_col = st.columns([1, 1.2])
    branch = b_col.selectbox(QT["branch"] if is_queue else "Cabang data sintetis", ["B", "C", "A"], key=lang_key("branch"),
                             format_func=(lambda b: QT["branches"][b]) if is_queue and lang == "EN" else BRANCH_LABEL.get)
    if is_queue:
        queue = r_col.radio(QT["route"], ["inspeksi", "cicilan"], horizontal=True, key=lang_key("queue"),
                            format_func=lambda q: {"inspeksi": QT["insp"], "cicilan": QT["inst"]}[q])
        l_col.radio(QT["lang"], ["ID", "EN"], horizontal=True, key="queue_lang")
branch_dir, ranked_path, features_path, demo_mode = sources(branch)
try:
    obs, ranked, F, reg = load(str(branch_dir), str(ranked_path), str(features_path))
except FileNotFoundError:
    st.error("Data demo belum tersedia. Jalankan perintah berikut di folder repo, lalu muat ulang halaman.")
    st.code("uv run tepat all", language="bash")
    st.stop()
store = case_store(demo_mode)
if store.mode == "sandbox":
    K.seed_examples(store, branch, ranked, load_truth(str(branch_dir)))
reports = report_store(store)
if store.mode == "sandbox":
    P.seed_examples(reports, branch, ranked)
lookback = 12
D = R.SCORING_DAY
st.sidebar.markdown('<div class="rail-bottom"><span class="rail-synthetic">SINTETIS</span>'
                    '<span class="rail-avatar" aria-label="Tim Johannes Kepler">JK</span></div>', unsafe_allow_html=True)

# The heading is written after loading so a case deep link can name the selected employer.
if view == "Kasus perusahaan":
    case_order = list(ranked.index)
    with case_slot, st.container(key="case_header_controls"):
        back, chooser, prev, nxt = st.columns([2.0, 1.0, .55, .55], gap="small", vertical_alignment="bottom")
        back.button("Daftar prioritas", key="case_back", on_click=lambda: st.session_state.update(view="Daftar prioritas"))
        with chooser.popover("Pilih"):
            eid = st.selectbox("Pilih perusahaan", case_order, key="case_employer",
                               format_func=lambda e: f"{reg.at[e, 'name']} (peringkat {int(ranked.at[e, 'rank'])})")
        # Read the selected value even when the popover is closed; it is a native selectbox in the widget tree.
        eid = st.session_state.get("case_employer", case_order[0])
        case_index = case_order.index(eid)
        prev.button("Sebelumnya", key="case_prev", disabled=case_index == 0,
                    on_click=lambda: st.session_state.update(case_employer=case_order[case_index - 1]))
        nxt.button("Berikutnya", key="case_next", disabled=case_index >= len(case_order) - 1,
                   on_click=lambda: st.session_state.update(case_employer=case_order[case_index + 1]))
with head_l:
    if view == "Kasus perusahaan":
        st.caption(f"Kasus perusahaan / peringkat {int(ranked.at[eid, 'rank'])} dari {ribu(len(ranked))}")
        st.title(reg.at[eid, "name"])
        st.caption(f"{SEKTOR[reg.at[eid, 'sector']]} di {reg.at[eid, 'kabkota']}. Nomor pemberi kerja {eid}.")
    else:
        st.title(view if not is_queue or lang == "ID" else QT["title"])
        st.caption(DESCRIPTIONS[view] if not is_queue else QT["sub"])

# 1. Inspection queue (design turn 3a) ------------------------------------------
if is_queue:
    L = ranked.join(reg[["name", "sector", "kabkota"]]).join(F[["band_obs", "hours"]])
    queue_panel.render(lang=lang, branch=branch, route=queue, L=L, F=F, truth=load_truth(str(branch_dir)),
                       rb=load_results()["branches"][branch], store=store, reasons=scorer.reasons, sektor=SEKTOR,
                       chart_config=CHART_CONFIG, open_case=open_case)

# 2. Case --------------------------------------------------------------------
if view == "Kasus perusahaan":
    case_panel.render(branch=branch, eid=eid, ranked=ranked, F=F, reg=reg, obs=obs, store=store, reports=reports,
                       branch_dir=branch_dir, sektor=SEKTOR, chart_config=CHART_CONFIG, open_case=open_case,
                       open_inbox=reports_panel.open_inbox, load_truth=load_truth, jt=jt, rp=rp, tanggal=tanggal,
                       waktu=waktu, flash=flash, lookback=lookback)

# 2a. Follow-up board ----------------------------------------------------------
if view == "Tindak lanjut":
    if store.mode == "sandbox":
        st.info("Versi daring: setiap pengunjung mendapat ruang kasus sendiri yang sudah berisi beberapa contoh. "
                "Perubahan Anda tidak terlihat oleh pengunjung lain dan hilang saat sesi ditutup. Tidak ada surat yang dikirim.")
    else:
        st.caption(f"Disimpan di {CASES_FILE.name} pada komputer ini. Tidak ada surat yang dikirim.")
        if store.count(branch) == 0 and st.button("Isi contoh kasus", key="fu_seed"):
            K.seed_examples(store, branch, ranked, load_truth(str(branch_dir)))
            st.rerun()
    C = store.cases(branch)
    sm = store.summary(branch)
    workspace_ui.kpis([
        ("Kasus terbuka", str(sm["open"]), f'{sm["overdue"]} lewat jadwal'),
        ("Kasus selesai", str(sm["closed"]), "Kasus yang sudah ditutup"),
        ("Pelanggaran terbukti", f'{sm["confirmed"]} dari {sm["closed"]}' if sm["closed"] else "-",
         "Termasuk koreksi mandiri dan cicilan"),
        ("Rp dipulihkan", jt(sm["recovered_rp"]), f'Perkiraan TEPAT {jt(sm["estimated_rp_closed"])}'),
    ], key="workspace_kpis_followup")
    if sm["closed"]:
        st.caption(f"Dari {sm['closed']} kasus selesai: {jt(sm['recovered_rp'])} dipulihkan, dibanding perkiraan TEPAT "
                   f"{jt(sm['estimated_rp_closed'])}. {sm['false_alarms']} kasus tanpa pelanggaran (alarm palsu). "
                   f"{sm['overdue']} kasus terbuka lewat jadwal.")
    if C.empty:
        st.info("Belum ada kasus. Mulai dari halaman Daftar prioritas atau Kasus perusahaan.")
    else:
        today = datetime.now().date().isoformat()
        C["name"] = C.employer_id.map(reg["name"])
        C["late"] = (C.status != K.CLOSED) & (C.due_date < today)
        quick_labels = {"Terbuka": int((C.status != K.CLOSED).sum()),
                        "Lewat jadwal": int(C.late.sum()), "Selesai": int((C.status == K.CLOSED).sum()),
                        "Semua": len(C)}
        with st.container(border=True, key="workspace_followup_board"):
            st.subheader("Kasus")
            quick = st.radio("Tampilkan kasus", list(quick_labels), horizontal=True, key="fu_quick_view",
                             format_func=lambda choice: f"{choice} {quick_labels[choice]}", label_visibility="collapsed")
            short = C if quick == "Semua" else C[C.late] if quick == "Lewat jadwal" else C[C.status == K.CLOSED] if quick == "Selesai" else C[C.status != K.CLOSED]
            short = short.assign(_closed=short.status == K.CLOSED).sort_values(["_closed", "due_date", "case_id"])
            st.markdown('<div class="ws-case-head"><span>Perusahaan</span><span>Status</span><span>Pemeriksa</span>'
                        '<span>Tanggal</span><span>Jadwal / hasil</span><span>Perkiraan</span><span>Dipulihkan</span>'
                        '<span>Sumber</span></div>', unsafe_allow_html=True)
            for _, item in short.head(14).iterrows():
                status = str(item.status)
                late = bool(item.late)
                row = '<div class="ws-case-row">' + ''.join([
                    f'<span class="ws-case-firm">{escape(str(item["name"]))}</span>',
                    f'<span>{workspace_ui.row_badge(status, "late" if late else "done" if status == K.CLOSED else "quiet")}</span>',
                    f'<span>{escape(str(item.owner))}</span>',
                    f'<span>{escape(tanggal(item.closed_at if status == K.CLOSED else item.due_date))}</span>',
                    f'<span>{escape(K.OUTCOMES.get(item.outcome, "Lewat jadwal" if late else "Sesuai jadwal"))}</span>',
                    f'<span>{escape(jt(item.est_rp))}</span>',
                    f'<span>{escape(jt(item.recovered_rp)) if pd.notna(item.recovered_rp) else "-"}</span>',
                    f'<span>{escape(SUMBER_KASUS.get(item.source, str(item.source)))}</span>']) + '</div>'
                with st.container(key=f"workspace_case_row_{int(item.case_id)}"):
                    st.markdown(row, unsafe_allow_html=True)
                    st.button(f"Buka {item['name']}", key=f"fu_row_{int(item.case_id)}",
                              on_click=open_case, args=(item.employer_id,))
            if short.empty:
                st.caption("Tidak ada kasus di filter ini. Mulai dari Daftar prioritas.")
            if len(short) > 14:
                st.caption(f"{len(short) - 14} kasus lain tersedia pada tabel lengkap di bawah.")
        with st.expander("Filter lanjutan dan tabel lengkap", expanded=False):
            st.caption("Gunakan filter untuk memeriksa semua kasus, termasuk yang tidak ditampilkan pada papan ringkas.")
            f1, f2, f3 = st.columns(3)
            pick_status = f1.multiselect("Status", K.STATUSES, default=K.STATUSES, key="fu_filter_status")
            pick_owner = f2.multiselect("Pemeriksa", K.INSPECTORS, default=K.INSPECTORS, key="fu_filter_owner")
            only_late = f3.checkbox("Hanya yang lewat jadwal", key="fu_filter_late")
        V = C[C.status.isin(pick_status) & C.owner.isin(pick_owner) & (C.late | ~only_late)].copy()
        V["order"] = np.where(V.status == K.CLOSED, 1, 0)
        V = V.sort_values(["order", "due_date", "case_id"])
        st.subheader("Kasus")
        st.dataframe(pd.DataFrame({
            "Perusahaan": V["name"], "Status": V.status, "Pemeriksa": V.owner,
            "Tindak lanjut": V.due_date.map(tanggal),
            "Jadwal": np.where(V.status == K.CLOSED, "-", np.where(V.late, "Lewat jadwal", "Sesuai jadwal")),
            "Saran sistem": V.action, "Perkiraan Rp": V.est_rp.map(jt),
            "Hasil": V.outcome.map(K.OUTCOMES).fillna("-"),
            "Rp dipulihkan": [jt(x) if pd.notna(x) else "-" for x in V.recovered_rp],
            "Sumber": V.source.map(SUMBER_KASUS)}), hide_index=True, width="stretch", height=min(460, 38 + 35 * len(V)),
            column_order=["Perusahaan", "Status", "Pemeriksa", "Tindak lanjut", "Jadwal", "Hasil", "Rp dipulihkan",
                          "Perkiraan Rp", "Saran sistem", "Sumber"],
            column_config={"Perusahaan": st.column_config.TextColumn(width="medium"),
                           "Status": st.column_config.TextColumn(width="medium"),
                           "Hasil": st.column_config.TextColumn(width="medium"),
                           "Saran sistem": st.column_config.TextColumn(width="medium")})
        if len(V):
            o1, o2 = st.columns([3, 1], gap="large", vertical_alignment="bottom")
            target = o1.selectbox("Pilih kasus", list(V.employer_id), key="fu_pick",
                                  format_func=lambda e: f"{reg.at[e, 'name']} ({C.set_index('employer_id').at[e, 'status']})")
            o2.button("Buka kasus", key="fu_open_case", type="primary", width="stretch", on_click=open_case, args=(target,))
        if (C.source == "contoh").any():
            st.caption("Baris bersumber Contoh demo dibuat otomatis agar papan ini tidak kosong. Hasil kasus contoh yang selesai "
                       "diambil dari kunci jawaban sintetis; di dunia nyata hasil hanya diketahui setelah pemeriksaan.")
    with st.container(border=True, key="glass_app_5"):
        st.subheader("Hasil sebagai data latih")
        labels = store.export_labels(branch)
        st.caption("Setiap kasus selesai menjadi satu label (1 = pelanggaran terbukti, 0 = tidak ditemukan). TEPAT tidak "
                   "melatih ulang model di aplikasi ini. Label dikumpulkan untuk pelatihan ulang berkala setelah ditinjau, "
                   "bersama hasil jalur acak agar model tidak hanya belajar dari kasus yang ia pilih sendiri.")
        st.download_button("Unduh label hasil (CSV)", labels.to_csv(index=False).encode("utf-8"),
                           file_name=f"tepat_label_hasil_{branch}.csv", mime="text/csv", key="fu_export",
                           disabled=labels.empty)

# Worker allegations inbox ----------------------------------------------------
if view == "Laporan pekerja":
    reports_panel.render(reports, branch, ranked, reg, open_case=open_case, waktu=waktu, legacy_path=TIPS)

# 2b. Second list: businesses missing from the BPJS Kesehatan registry ------------------
if view == "Usaha belum terdaftar":
    up, tp = unreg_sources(branch, demo_mode)
    if not (up.is_file() and tp.is_file()):
        st.info("Daftar kedua belum dibuat. Jalankan perintah berikut di folder repo, lalu muat ulang halaman.")
        st.code("uv run tepat unreg", language="bash")
        st.stop()
    U, UT = load_unreg(str(up), str(tp))
    if branch == "A":
        st.warning("Cabang A dipakai untuk melatih model daftar ini. Angka uji berasal dari cabang B atau C.")
    unregistered_panel.render(branch=branch, U=U, UT=UT, sektor=SEKTOR, skala=SKALA, results_dir=RESULTS,
                              jt=jt, rp=rp, ribu=ribu, angka=angka, kali=kali, pct_range=pct_range, load_csv=load_csv)

# 3. Benchmark ---------------------------------------------------------------
if view == "Benchmark":
    rb = load_results()["branches"][branch]
    final = rb.get("final_ranking", "tepat_v2_1")
    best = rb["rankings"][final]
    baseline = rb["rankings"]["rules_only"]
    if branch == "A":
        st.warning("Cabang A dipakai untuk tuning. Angka utama harus berasal dari cabang B atau C.")
    workspace_ui.kpis([
        ("Rp per jam pemeriksaan", jt(best["rp_per_hour"]), "TEPAT v2.1, seed bawaan"),
        ("Presisi pemeriksaan", f'{best["precision"]:.0%}', f'Aturan saja {baseline["precision"]:.0%}'),
        ("Perusahaan jujur ditandai", f'{angka(best["honest_flag_rate"] * 100)}%',
         f'Aturan saja {angka(baseline["honest_flag_rate"] * 100)}%'),
        ("Dibanding aturan saja", kali(best["rp_per_hour"] / max(1, baseline["rp_per_hour"])),
         "Rp per jam pada seed bawaan, bukan dampak nasional"),
    ], key="workspace_kpis_benchmark")
    st.caption(f"Cabang {branch}, seed bawaan: {ribu(rb['employers'])} perusahaan; {rb['violators']} pelanggar ditanam "
               f"({rb['prevalence']:.0%}); anggaran {rb['budget_hours']:.0f} jam, sekitar 10% total jam. "
               "Data sintetis, bukan bukti dampak operasional BPJS atau estimasi nasional. Rentang lima seed ada di bawah.")
    rows = []
    for k, v in rb["rankings"].items():
        rows.append(dict(Peringkat=RANK_LABEL.get(k, k), key=k, rp_per_jam=v["rp_per_hour"] / 1e6, presisi=v["precision"],
                         recall=v["recall"], jujur_ditandai=v["honest_flag_rate"], diperiksa=v["inspected"], rp_total=v["rp_recovered"] / 1e6))
    B = pd.DataFrame(rows)
    B["kelompok"] = np.where(B.key == final, "TEPAT utama", np.where(B.key.str.startswith("tepat"), "TEPAT lainnya", "Pembanding"))
    x1, x2 = st.columns(2, gap="medium")
    with x1, st.container(border=True, key="workspace_benchmark_rph"):
        st.subheader("Rupiah per jam")
        st.caption("Rp juta / jam pemeriksaan, seed bawaan sintetis")
        ranked_rp = B.sort_values("rp_per_jam", ascending=False)
        st.markdown(workspace_ui.bar_rows([
            (row.Peringkat, float(row.rp_per_jam), angka(float(row.rp_per_jam), 2),
             "tepat" if row.key == final else "other" if row.key.startswith("tepat") else "base")
            for row in ranked_rp.itertuples()], maximum=float(B.rp_per_jam.max())), unsafe_allow_html=True)
    with x2, st.container(border=True, key="workspace_benchmark_precision"):
        st.subheader("Presisi pemeriksaan")
        st.caption("Bagian pemeriksaan yang menemukan pelanggaran, seed bawaan sintetis")
        ranked_precision = B.sort_values("presisi", ascending=False)
        st.markdown(workspace_ui.bar_rows([
            (row.Peringkat, float(row.presisi), f"{float(row.presisi):.0%}",
             "tepat" if row.key == final else "other" if row.key.startswith("tepat") else "base")
            for row in ranked_precision.itertuples()], maximum=1.0), unsafe_allow_html=True)
    with st.container(border=True, key="workspace_benchmark_table"):
        st.subheader("Hasil lengkap")
        table_rows = ''.join(
            '<tr' + (' class="ws-table-highlight"' if row.key == final else '') + '>'
            f'<th>{escape(row.Peringkat)}</th><td>{angka(row.rp_per_jam, 2)}</td>'
            f'<td>{row.presisi:.0%}</td><td>{row.recall:.0%}</td>'
            f'<td>{angka(row.jujur_ditandai * 100)}%</td><td>{ribu(row.diperiksa)}</td>'
            f'<td>{angka(row.rp_total, 0)}</td></tr>' for row in B.sort_values('rp_per_jam', ascending=False).itertuples())
        st.markdown('<div class="ws-table-scroll"><table class="ws-table"><thead><tr>'
                    '<th>Metode</th><th>Rp jt/jam</th><th>Presisi</th><th>Recall</th>'
                    '<th>Jujur ditandai</th><th>Diperiksa</th><th>Rp jt total</th>'
                    '</tr></thead><tbody>' + table_rows + '</tbody></table></div>', unsafe_allow_html=True)
    table = B.drop(columns=["key", "kelompok"]).rename(columns={"Peringkat": "Metode", "rp_per_jam": "Rp jt/jam",
        "presisi": "Presisi", "recall": "Recall", "jujur_ditandai": "Jujur ditandai", "diperiksa": "Diperiksa", "rp_total": "Rp jt total"})
    with st.expander("Tabel hasil untuk disalin"):
        st.dataframe(table.style.format({"Rp jt/jam": lambda x: angka(x, 2), "Presisi": "{:.0%}", "Recall": "{:.0%}",
                                         "Jujur ditandai": lambda x: f"{angka(x * 100)}%", "Diperiksa": lambda x: angka(x, 0),
                                         "Rp jt total": lambda x: angka(x, 0)}), width="stretch", hide_index=True)
    st.subheader("Ketahanan lintas seed")
    ss = load_csv(str(RESULTS / "sweep" / "seeds_summary.csv"))
    ss = ss[ss.branch == branch].set_index("ranking")
    sr = load_csv(str(RESULTS / "sweep" / "seeds_ratios.csv")).set_index("branch").loc[branch]
    v = ss.loc[final]
    with st.container(key="summary_seeds"):
        a, b, c, d = st.columns(4)
        a.metric("Rp per jam, lima seed", jt(v.rp_per_hour_juta_mean * 1e6),
                 help=f"{angka(v.rp_per_hour_juta_min)} s.d. {angka(v.rp_per_hour_juta_max)} jt per jam")
        b.metric("Presisi, lima seed", f"{v.precision_mean:.0%}", help=f"{v.precision_min:.0%} s.d. {v.precision_max:.0%}")
        c.metric("Dibanding aturan saja", kali(sr.x_rules_of_means), help=f"Per seed {kali(sr.x_rules_min)} s.d. {kali(sr.x_rules_max)}")
        d.metric("Dibanding acak", kali(sr.x_random_of_means), help=f"Per seed {kali(sr.x_random_min)} s.d. {kali(sr.x_random_max)}")
    st.caption(f"Lima seed, data dibangkitkan ulang tiap seed; model dilatih ulang di cabang A. {RANK_LABEL[final]}: "
               f"{angka(v.rp_per_hour_juta_mean)} jt per jam ({angka(v.rp_per_hour_juta_min)} s.d. {angka(v.rp_per_hour_juta_max)}), "
               f"{kali(sr.x_rules_of_means)} aturan saja (per seed {kali(sr.x_rules_min)} s.d. {kali(sr.x_rules_max)}). "
               "Rasio dihitung dari rata-rata, bukan rata-rata rasio per seed.")
    seed_rows = [{"Metode": RANK_LABEL.get(k, k),
                  "Rp jt/jam": f"{angka(r.rp_per_hour_juta_mean)} ({angka(r.rp_per_hour_juta_min)} s.d. {angka(r.rp_per_hour_juta_max)})",
                  "Presisi": pct_range(r.precision_mean, r.precision_min, r.precision_max),
                  "Recall": pct_range(r.recall_mean, r.recall_min, r.recall_max),
                  "Jujur ditandai": f"{r.honest_flag_rate_mean:.1%}".replace(".", ",")} for k, r in ss.iterrows()]
    st.dataframe(pd.DataFrame(seed_rows), width="stretch", hide_index=True)
    if branch in ("B", "C"):
        with st.expander("Kapan keunggulan mengecil? Uji sensitivitas"):
            sens = load_csv(str(RESULTS / "sweep" / "sensitivity.csv"))
            sens = sens[sens.branch == branch]
            nilai = {"5%": "5%", "10%": "10%", "35%": "35%", "20%": "20%", "6": "6", "24": "24",
                     "flat 6/6/6/6": "rata 6/6/6/6", "steep 2/6/16/48": "curam 2/6/16/48"}
            st.dataframe(pd.DataFrame({
                "Asumsi": sens.dial.map(DIAL),
                "Nilai": [("semua bawaan" if d_ == "default" else nilai.get(str(x), str(x)))
                          for d_, x in zip(sens.dial, sens.value)],
                "TEPAT Rp jt/jam": sens.v21_rp_hour.map(angka), "Aturan Rp jt/jam": sens.rules_rp_hour.map(angka),
                "Dibanding aturan": sens.x_rules_of_means.map(kali), "Presisi TEPAT": sens.v21_precision.map("{:.0%}".format),
                "Jujur ditandai TEPAT": sens.v21_honest_flag.map(lambda x: f"{x:.1%}".replace(".", ",")),
                "Jujur ditandai aturan": sens.rules_honest_flag.map(lambda x: f"{x:.1%}".replace(".", ","))}),
                width="stretch", hide_index=True)
            low = sens[(sens.dial == "prevalence") & (sens.value == "5%")]
            if len(low):
                lo = low.iloc[0]
                st.caption(f"Tiga seed per baris, satu asumsi diubah tiap kali dari nilai bawaan (prevalensi 20%, masa tagih 12 bulan, "
                           f"anggaran 10%, jam 3/6/12/24 per skala). Keunggulan paling kecil saat jam pemeriksaan rata atau "
                           f"anggaran besar. Pada prevalensi 5%, presisi TEPAT turun ke {lo.v21_precision:.0%} dan TEPAT menandai "
                           f"perusahaan jujur {angka(lo.v21_honest_flag * 100)}% dibanding aturan {angka(lo.rules_honest_flag * 100)}%, "
                           "karena anggaran tidak bisa diisi pelanggar saja.")
    with st.expander("Di mana model masih lemah? Recall dan look-alike"):
        y1, y2 = st.columns(2, gap="large")
        with y1:
            st.subheader("Recall per varian pelanggaran")
            rv = pd.DataFrame({"Aturan saja": baseline["recall_by_variant"], "TEPAT": best["recall_by_variant"]})
            st.dataframe(rv.style.format("{:.0%}"), width="stretch")
        with y2:
            st.subheader("Look-alike jujur yang ditandai")
            lv = pd.DataFrame({"Aturan saja": baseline["lookalike_flag_rate"], "TEPAT": best["lookalike_flag_rate"]})
            st.dataframe(lv.style.format("{:.0%}"), width="stretch")
        cov = rb.get("rp_range_coverage")
        if cov is not None:
            st.caption(f"Kalibrasi rentang Rp: {cov:.0%} pelanggar terpilih berada dalam rentang prediksi. Target sekitar 80% (persentil 10-90).")
    T = load_truth(str(branch_dir))
    rules = scorer.rules_only(F)
    honest = T[~T.is_violator & T.lookalikes.str.contains("C1")].index
    ex = [e for e in rules.index[:200] if e in honest]
    if ex:
        e = ex[0]
        st.success(f"**Contoh perusahaan jujur: {reg.at[e, 'name']}**. Aturan sederhana menandainya di peringkat "
                   f"#{list(rules.index).index(e) + 1} karena TK > Kes. TEPAT mengenali pekerja proyek (`project`) "
                   f"dan menempatkannya di #{int(ranked.at[e, 'rank'])}. Kunci jawaban sintetis: jujur, look-alike C1.")
    st.caption("Batasan: antrean utama hanya memuat pemberi kerja yang sudah terdaftar di BPJS Kesehatan; usaha yang belum terdaftar "
               "diuji terpisah di halaman Usaha belum terdaftar. Prevalensi, jam pemeriksaan, dan masa tagih merupakan asumsi simulasi; "
               "pengaruhnya ada di uji sensitivitas. Recall per varian dan look-alike di atas berasal dari seed bawaan. "
               "Tingkat kematangan: prototipe.")

# 4. Fairness ----------------------------------------------------------------
if view == "Keadilan & jalur acak":
    fairness_panel.render(branch, RESULTS / "fairness", load_results()["branches"][branch], SEKTOR, CHART_CONFIG)

if view == "Cek kepesertaan saya":
    st.caption("Untuk pekerja. Di dunia nyata fitur ini berada di Mobile JKN; di sini memakai data sintetis dan NIK contoh diawali 99.")
    if msg := st.session_state.pop(f"worker_report_flash_{branch}", None):
        st.success(msg)
    if problem := st.session_state.pop(f"worker_report_error_{branch}", None):
        st.error(problem)
    ks, tk, bill, mem = obs["ks_membership"], obs["tk_exchange"], obs["billing"], obs["ks_members"]

    @st.cache_data
    def demo_niks(branch_dir: str) -> dict:
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

    demos = demo_niks(str(branch_dir))
    with st.container(border=True, key="workspace_worker_card"):
        pick = st.radio("Contoh NIK sintetis", list(demos), horizontal=True, key="worker_scenario")
        nik = st.text_input("NIK (16 digit)", value=demos.get(pick, ""), max_chars=16, key=f"worker_nik_{branch}_{pick}")
        if nik:
            my_ks = ks[ks.nik == nik].sort_values("ppu_start")
            my_tk = tk[(tk.nik == nik) & (tk.quarter == tk[tk.nik == nik].quarter.max())] if (tk.nik == nik).any() else tk.iloc[0:0]
            my_dep = mem[(mem.nik == nik) & (mem.segment == "DEPENDENT")]
            active = my_ks[my_ks.ppu_end.isna() | (my_ks.ppu_end > D)]
            alerts = []
            with st.container(key="summary"):
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
                st.error(text)
                if st.button("Kirim laporan demo", key=f"lapor_{branch}_{emp}_{jenis}_{i}", type="primary"):
                    try:
                        receipt = reports.submit(branch, P.hash_reporter(nik), emp, jenis)
                    except P.ReportError as e:
                        st.session_state[f"worker_report_error_{branch}"] = str(e)
                    else:
                        acknowledgement = ("Pengiriman duplikat dicatat pada tiket yang sama" if receipt.duplicate
                                           else "Laporan demo baru dicatat")
                        st.session_state[f"worker_report_flash_{branch}"] = (
                            f"{acknowledgement}. Nomor tiket **{receipt.ticket}**. "
                            "Dapat ditinjau di halaman Laporan pekerja. Tidak membuat kasus atau mengubah peringkat secara otomatis.")
                    st.rerun()
            if len(my_ks) == 0 and len(my_tk) == 0 and len(my_dep) == 0:
                st.info("NIK tidak ditemukan" + (" dalam data pekerja versi daring" if demo_mode else "")
                        + ". Pilih skenario demo untuk mencoba data yang tersedia.")
    st.caption("Triase laporan tersedia di halaman Laporan pekerja. Notifikasi proaktif lewat Mobile JKN masih berupa rancangan; tidak ada pengiriman dari prototipe.")
    if demo_mode:
        n_detail = demo_worker_employers(branch)
        st.caption(f"Versi daring ini memuat data pekerja dari {n_detail} perusahaan contoh, bukan {ribu(len(reg))}. "
                   "Hasil untuk setiap NIK yang dimuat tetap sama dengan data lengkap. Halaman lain memakai semua perusahaan.")
    reports_panel.limits(reports)

st.markdown('<footer class="app-footer"><span>TEPAT / Tim Johannes Kepler / Healthkathon 2026</span>'
            '<span>Data sintetis. Bukan data BPJS asli. Keputusan pemeriksaan tetap pada petugas.</span></footer>', unsafe_allow_html=True)
