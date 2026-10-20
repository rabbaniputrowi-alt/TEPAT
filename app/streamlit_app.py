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
import csv
import hashlib
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
from tepat import rules as R  # noqa: E402
from tepat import scorer  # noqa: E402
if str(ROOT / "app") not in sys.path:  # drafting_panel.py sits next to this script
    sys.path.insert(0, str(ROOT / "app"))
import drafting_panel  # noqa: E402

DATA = Path(os.environ.get("TEPAT_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("TEPAT_RESULTS", ROOT / "results"))
DEMO = ROOT / "demo"
TIPS = RESULTS / "tips.csv"
CASES_FILE = RESULTS / "cases.sqlite"
SUMBER_KASUS = {"contoh": "Contoh demo", "petugas": "Petugas", "massal": "Mulai massal"}
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
    return json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))


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
css = (ASSETS / "workspace.css").read_text(encoding="utf-8").replace("__PUBLIC_SANS__", font_data)
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
VIEWS = ["Daftar prioritas", "Kasus perusahaan", "Tindak lanjut", "Usaha belum terdaftar", "Benchmark", "Keadilan & jalur acak",
         "Cek kepesertaan saya"]
DESCRIPTIONS = {
    "Daftar prioritas": "Tentukan kapasitas pemeriksaan. Dahulukan potensi pemulihan per jam tertinggi.",
    "Kasus perusahaan": "Telusuri sinyal, dasar hukum, dan dokumen sebelum mengambil keputusan.",
    "Tindak lanjut": "Pantau kasus yang sedang ditangani, jadwal berikutnya, dan hasil pemeriksaan.",
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
st.sidebar.markdown('<div class="sidebar-brand">TEPAT</div>', unsafe_allow_html=True)
view = st.sidebar.radio("Ruang kerja", VIEWS, key="view", label_visibility="collapsed")
branch = st.sidebar.selectbox("Cabang data sintetis", ["B", "C", "A"], key="branch",
                              format_func=lambda b: {"A": "A (tuning)", "B": "B (uji tertahan)", "C": "C (uji bergeser)"}[b])
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
lookback = 12
D = R.SCORING_DAY
st.sidebar.markdown(f'<div class="sidebar-context"><b>{ribu(len(reg))}</b> perusahaan sintetis<br>'
                    f'{month_label(0)} s.d. {month_label(R.HORIZON_MONTHS - 1)}<br>Model TEPAT v2.1</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-note"><strong>Bukan data BPJS asli</strong>'
                    'Perusahaan, pekerja, NIK, dan UMK fiktif. NIK demo diawali 99. Tidak terhubung ke layanan BPJS.</div>'
                    '<div class="sidebar-credit">Tim Johannes Kepler<br>Healthkathon 2026</div>', unsafe_allow_html=True)
st.title(view)
st.caption(DESCRIPTIONS[view])

# 1. Inspection queue --------------------------------------------------------
if view == "Daftar prioritas":
    L = ranked.join(reg[["name", "sector", "kabkota"]]).join(F[["band_obs", "hours"]])
    summary = st.container(key="summary")
    with st.container(border=True, key="capacity"):
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("Kapasitas pemeriksaan")
            total_h = float(F.hours.sum())
            budget = st.slider("Anggaran jam pemeriksaan", 50, int(total_h * 0.3), int(total_h * 0.10), step=25,
                               key="budget", help="Anggaran seluruh cabang, bukan hanya baris setelah filter.")
        with right:
            st.subheader("Rute tindak lanjut")
            queue = st.radio("Antrean", ["inspeksi", "cicilan"], horizontal=True, key="queue",
                             format_func=lambda q: {"inspeksi": "Pemeriksaan", "cicilan": "Cicilan"}[q], label_visibility="collapsed")
            st.caption("Ketidakmampuan bayar diarahkan ke cicilan, bukan otomatis ke pemeriksaan.")
    insp = L[L.queue == "inspeksi"]
    in_budget = set(insp.index[insp.hours.cumsum() <= budget])
    sel = L.loc[list(in_budget)]
    with summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Perkiraan pemulihan", jt(sel.expected_rp.sum()))
        m2.metric("Perusahaan dalam anggaran", f"{len(sel)}")
        m3.metric("Perkiraan Rp per jam", jt(sel.expected_rp.sum() / max(1, sel.hours.sum())))
        m4.metric("Antrean cicilan", f"{int((L.queue == 'cicilan').sum())}")
    fresh = [e for e in insp.index if e in in_budget and store.get(branch, e) is None]
    bulk_left, bulk_right = st.columns([3, 1], gap="large", vertical_alignment="center")
    bulk_left.caption(f"{len(fresh)} dari {len(in_budget)} perusahaan dalam anggaran belum punya tindak lanjut. "
                      "Memulai tindak lanjut hanya membuat kasus berstatus Baru; tidak ada surat yang dikirim.")
    if bulk_right.button("Mulai tindak lanjut", key="bulk_follow_up", disabled=not fresh, width="stretch"):
        created, skipped = store.bulk_open(branch, L.loc[fresh, ["est_rp", "p_violation", "action"]])
        st.session_state["bulk_flash"] = f"{created} kasus baru dibuat. Lihat halaman Tindak lanjut."
        st.rerun()
    if msg := st.session_state.pop("bulk_flash", None):
        st.success(msg)
    with st.expander("Filter daftar", expanded=False):
        c1, c2 = st.columns(2)
        kab = c1.multiselect("Kab/Kota", sorted(L.kabkota.unique()), default=sorted(L.kabkota.unique()), key="districts")
        tiers = c2.multiselect("Tingkat", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang"], key="tiers")
    V = L[L.kabkota.isin(kab) & L.tier.isin(tiers) & (L.queue == queue)].copy()
    V["anggaran"] = np.where(V.index.isin(in_budget), "Terpilih", "Di luar anggaran")
    st.subheader("Perusahaan untuk ditinjau")
    st.caption(f"{len(V)} perusahaan ditampilkan. Ringkasan anggaran mencakup seluruh cabang; filter hanya mengubah daftar.")
    show = pd.DataFrame({
        "Peringkat": V["rank"], "Anggaran": V.anggaran, "Perusahaan": V["name"], "Sektor": V.sector.map(SEKTOR),
        "Kab/Kota": V.kabkota, "Skala": V.band_obs, "Tingkat": V.tier,
        "P(pelanggaran)": V.p_violation.round(2), "Perkiraan Rp": V.est_rp.map(jt),
        "Rentang": [f"{jt(a)} s.d. {jt(b)}" for a, b in zip(V.rp_low, V.rp_high)],
        "Rp/jam": (V.expected_rp / V.hours).map(jt), "Alasan utama": V.top_reasons.str.split(" \\| ").str[0],
        "Tindakan": V.action}, index=V.index)
    table_col, preview_col = st.columns([3, 1], gap="large")
    with table_col:
        st.dataframe(show, height=460, width="stretch", hide_index=True,
                     column_order=["Peringkat", "Perusahaan", "P(pelanggaran)", "Rp/jam", "Tingkat", "Anggaran",
                                   "Kab/Kota", "Sektor", "Skala", "Perkiraan Rp", "Rentang", "Alasan utama", "Tindakan"],
                     column_config={"Peringkat": st.column_config.NumberColumn("#", width="small"),
                                    "Perusahaan": st.column_config.TextColumn(width="medium"),
                                    "P(pelanggaran)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    with preview_col, st.container(border=True):
        if not V.empty:
            top_id = V.index[0]
            top = V.iloc[0]
            st.caption("Kasus teratas dari filter")
            st.markdown(f'<div class="case-preview-name">{escape(top["name"])}</div>'
                        f'<div class="case-preview-detail">Peringkat {int(top["rank"])}, tingkat {escape(top.tier.lower())}.<br>'
                        f'{escape(SEKTOR[top.sector])} di {escape(top.kabkota)}.</div>', unsafe_allow_html=True)
            st.metric("P(pelanggaran)", f"{top.p_violation:.0%}")
            st.caption(top.action)
            st.button("Buka kasus", key="open_case", type="primary", width="stretch", on_click=open_case, args=(top_id,))
        else:
            st.info("Tidak ada perusahaan sesuai filter. Pilih wilayah atau tingkat lain pada Filter daftar.")
    st.caption("Petugas tetap mengambil keputusan. Tidak ada keputusan otomatis semata (UU PDP Ps. 10). "
               "Jalur acak 8% ditampilkan pada halaman Keadilan; slider ini mengilustrasikan prioritas tanpa jalur acak.")

# 2. Case --------------------------------------------------------------------
if view == "Kasus perusahaan":
    tips = read_tips()
    order = list(ranked.index)
    eid = st.selectbox("Pilih perusahaan", order, index=0, key="case_employer",
                       format_func=lambda e: f"{reg.at[e, 'name']} (peringkat {int(ranked.at[e, 'rank'])})")
    r, f_ = ranked.loc[eid], F.loc[eid]
    st.subheader(reg.at[eid, "name"])
    st.caption(f"{SEKTOR[reg.at[eid, 'sector']]} di {reg.at[eid, 'kabkota']}. Nomor pemberi kerja {eid}.")
    with st.container(key="summary"):
        a, b, c, d = st.columns(4)
        a.metric("Perkiraan pemulihan", jt(r.est_rp), help=f"Rentang {jt(r.rp_low)} s.d. {jt(r.rp_high)} (persentil 10-90 dari data latih)")
        b.metric("Peringkat", f"#{int(r['rank'])}")
        c.metric("P(pelanggaran)", f"{r.p_violation:.0%}")
        d.metric("Waktu pemeriksaan", f"{int(f_.hours)} jam")
    with st.container(key="recommended"):
        st.info(f"**Tindakan yang disarankan:** {r.action}" + (f"  \n**Catatan pembayaran:** {r.pay_label}" if r.pay_label else ""))
    n_tip = int((tips.employer_id == eid).sum()) if len(tips) else 0
    if n_tip:
        st.warning(f"{n_tip} laporan demo pekerja masuk melalui Cek Kepesertaan Saya untuk perusahaan ini.")
    follow_up_panel(store, branch, eid, r)
    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Mengapa perusahaan ini?")
            rs = scorer.reasons(f_)
            if not rs:
                st.success("Tidak ada sinyal pelanggaran. Jalur hijau.")
            for _, text, basis in rs[:6]:
                st.markdown(f'<div class="signal-row"><strong>{escape(text)}</strong>'
                            f'<small>Dasar: {escape(basis)}</small></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.subheader("Dokumen pemeriksaan")
            documents = "".join(f"<li>{escape(x)}</li>" for x in scorer.brief(f_))
            st.markdown(f'<ul class="document-list">{documents}</ul>', unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.subheader("Komposisi perkiraan Rp")
            comp = pd.DataFrame({"Komponen": ["Pekerja tidak terdaftar", "Upah (TK > Kes)", "Upah (pola sejenis)", "Tunggakan iuran",
                                              "Terlambat daftar", "Nonaktif sebelum keluar", "Pindah segmen", "Klinik perusahaan",
                                              "PHK berulang", "Biaya rawat saat nonaktif", "Tanggungan PHK"],
                                 "Rp": [f_.rp_headcount * lookback, f_.rp_wage2c * lookback, f_.rp_wage_peer * lookback,
                                        f_.rp_arrears * lookback, f_.rp_lag * lookback, f_.rp_onoff * lookback,
                                        f_.rp_group_move * lookback, f_.rp_clinic * lookback, f_.rp_phk_gap * lookback,
                                        f_.rp_care_liability, f_.rp_phk_cover]})
            comp = comp[comp.Rp > 0]
            if len(comp):
                st.altair_chart(alt.Chart(comp).mark_bar(color="#087f82", cornerRadiusEnd=4, height=20).encode(
                    x=alt.X("Rp:Q", title="Rp (12 bulan)", axis=alt.Axis(format="~s")),
                    y=alt.Y("Komponen:N", sort="-x", title=None),
                    tooltip=["Komponen", alt.Tooltip("Rp:Q", format=",.0f")]).properties(height=max(130, len(comp) * 32)).configure(**CHART_CONFIG),
                    theme=None, width="stretch")
            else:
                st.caption("Tidak ada komponen Rp yang teridentifikasi.")
        with st.container(border=True):
            bill = obs["billing"][obs["billing"].employer_id == eid].copy()
            bill["Bulan"] = bill.month.map(month_label)
            bill["Badan"] = bill.agency.map({"KS": "BPJS Kesehatan", "TK": "BPJS Ketenagakerjaan"})
            bill["Terlambat (hari)"] = bill.days_late
            bill["Kurang bayar"] = (bill.billed_rp - bill.paid_rp).clip(lower=0)
            st.subheader("Pembayaran iuran 24 bulan")
            st.altair_chart(alt.Chart(bill).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X("month:Q", title="Bulan ke- (0 = Okt 2024)", axis=alt.Axis(tickCount=6)),
                y=alt.Y("Terlambat (hari):Q", title="Hari keterlambatan"),
                color=alt.Color("Badan:N", scale=alt.Scale(range=["#087f82", "#d48b4a"]), legend=alt.Legend(orient="bottom", title=None)),
                tooltip=["Bulan", "Badan", "Terlambat (hari)", alt.Tooltip("Kurang bayar:Q", format=",.0f")]).properties(height=210).configure(**CHART_CONFIG),
                theme=None, width="stretch")
    # Draf dokumen (feature/drafting): evidence-only drafts with human approval; nothing is sent.
    # HOOK(case-follow-up): record the approval in case history here. Today it only stores it in session state.
    def on_draft_approved(employer_id: str, doc_type: str, text: str) -> None:
        st.session_state.setdefault("approved_drafts", {})[f"{employer_id}:{doc_type}"] = {
            "employer_id": employer_id, "doc_type": doc_type, "text": text,
            "approved_at": datetime.now().isoformat(timespec="seconds")}
    drafting_panel.render(eid, r, f_, reg.loc[eid], SEKTOR[reg.at[eid, "sector"]], on_approved=on_draft_approved)
    with st.expander("Kunci jawaban (khusus demo sintetis)"):
        st.caption("Tidak tersedia bagi petugas di dunia nyata. Tidak dipakai untuk menentukan peringkat.")
        t = load_truth(str(branch_dir)).loc[eid]
        st.write({"melanggar": bool(t.is_violator), "varian yang ditanam": t.variants or "-",
                  "look-alike jujur": t.lookalikes or "-", "Rp dapat dipulihkan (sebenarnya)": rp(t.recoverable_rp_true)})

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
    with st.container(key="summary"):
        a, b, c, d = st.columns(4)
        a.metric("Kasus terbuka", f"{sm['open']}", help=f"{sm['overdue']} lewat jadwal tindak lanjut")
        b.metric("Kasus selesai", f"{sm['closed']}")
        c.metric("Pelanggaran terbukti", f"{sm['confirmed']} dari {sm['closed']}" if sm["closed"] else "-",
                 help="Pelanggaran terbukti, dikoreksi mandiri, atau masuk cicilan")
        d.metric("Rp dipulihkan", jt(sm["recovered_rp"]), help="Jumlah dari kasus yang sudah selesai")
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
        with st.expander("Filter kasus", expanded=False):
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
    with st.container(border=True):
        st.subheader("Hasil sebagai data latih")
        labels = store.export_labels(branch)
        st.caption("Setiap kasus selesai menjadi satu label (1 = pelanggaran terbukti, 0 = tidak ditemukan). TEPAT tidak "
                   "melatih ulang model di aplikasi ini. Label dikumpulkan untuk pelatihan ulang berkala setelah ditinjau, "
                   "bersama hasil jalur acak agar model tidak hanya belajar dari kasus yang ia pilih sendiri.")
        st.download_button("Unduh label hasil (CSV)", labels.to_csv(index=False).encode("utf-8"),
                           file_name=f"tepat_label_hasil_{branch}.csv", mime="text/csv", key="fu_export",
                           disabled=labels.empty)

# 2b. Second list: businesses missing from the BPJS Kesehatan registry ------------------
if view == "Usaha belum terdaftar":
    up, tp = unreg_sources(branch, demo_mode)
    if not (up.is_file() and tp.is_file()):
        st.info("Daftar kedua belum dibuat. Jalankan perintah berikut di folder repo, lalu muat ulang halaman.")
        st.code("uv run tepat unreg", language="bash")
        st.stop()
    U, UT = load_unreg(str(up), str(tp))
    st.info("Antrean utama hanya berisi pemberi kerja yang sudah terdaftar di BPJS Kesehatan. Daftar kedua ini memeriksa "
            "pemegang NIB di OSS dan usaha yang terdaftar di BPJS Ketenagakerjaan, tetapi tidak ada di registri BPJS Kesehatan. "
            "Antrean utama dan angka benchmark utama tidak berubah karena daftar ini.")
    if branch == "A":
        st.warning("Cabang A dipakai untuk melatih model daftar ini. Angka uji berasal dari cabang B atau C.")
    summary = st.container(key="summary")
    with st.container(border=True, key="unreg_capacity"):
        total_h = float(U.hours.sum())
        budget = st.slider("Anggaran jam pemeriksaan", 25, int(total_h * 0.3), int(total_h * 0.10), step=25, key="unreg_budget",
                           help="Bawaan sekitar 10% total jam kandidat, sama seperti pengujian.")
    U = U.assign(expected_rp=U.p_unregistered * U.est_rp.clip(lower=0))
    in_budget = set(U.index[U.hours.cumsum() <= budget])
    sel = U.loc[list(in_budget)]
    with summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Perkiraan iuran 12 bulan", jt(sel.expected_rp.sum()), help="Tertimbang P. Upah memakai UMK sebagai proksi.")
        m2.metric("Usaha dalam anggaran", f"{len(sel)}")
        m3.metric("Perkiraan Rp per jam", jt(sel.expected_rp.sum() / max(1, sel.hours.sum())))
        m4.metric("Pemeriksaan bersama TK", f"{int(sel.action.str.startswith('Pemeriksaan bersama').sum())}")
    st.subheader("Usaha untuk ditinjau")
    st.caption(f"{ribu(len(U))} kandidat dari OSS dan BPJS Ketenagakerjaan yang tidak ada di registri BPJS Kesehatan cabang ini.")
    show = pd.DataFrame({
        "Peringkat": U["rank"], "Anggaran": np.where(U.index.isin(in_budget), "Terpilih", "Di luar anggaran"),
        "Usaha": U["name"], "P(wajib, belum terdaftar)": U.p_unregistered.round(2),
        "Rp/jam": (U.expected_rp / U.hours).map(jt), "Perkiraan pekerja": U.est_workers.astype(int),
        "Sumber": np.where(U.tk_n > 0, "BPJS TK dan OSS", "OSS saja"), "Kab/Kota": U.kabkota,
        "Sektor": U.sector.map(SEKTOR), "Skala": U.skala_usaha.map(SKALA), "Tindakan": U.action,
        "Alasan utama": U.reasons.str.split(" \\| ").str[0]}, index=U.index)
    st.dataframe(show, height=420, width="stretch", hide_index=True,
                 column_config={"Peringkat": st.column_config.NumberColumn("#", width="small"),
                                "Usaha": st.column_config.TextColumn(width="medium"),
                                "P(wajib, belum terdaftar)": st.column_config.ProgressColumn(min_value=0, max_value=1, format="percent")})
    left, right = st.columns([3, 2], gap="large")
    with left, st.container(border=True):
        nib = st.selectbox("Pilih usaha", list(U.index), index=0, key="unreg_pick",
                           format_func=lambda n: f"{U.at[n, 'name']} (peringkat {int(U.at[n, 'rank'])})")
        u = U.loc[nib]
        st.subheader(u["name"])
        st.caption(f"{SEKTOR.get(u.sector, u.sector)} di {u.kabkota}. Skala {SKALA.get(u.skala_usaha, u.skala_usaha).lower()}. "
                   f"NIB {nib} (sintetis).")
        st.info(f"**Tindakan yang disarankan:** {u.action}")
        for text in u.reasons.split(" | "):
            st.markdown(f'<div class="signal-row"><strong>{escape(text)}</strong>'
                        f'<small>Dasar: {escape(u.legal_basis)}</small></div>', unsafe_allow_html=True)
    with right, st.container(border=True):
        st.subheader("Perkiraan nilai")
        st.metric("Perkiraan iuran 12 bulan", jt(u.est_rp))
        st.metric("P(wajib, belum terdaftar)", f"{u.p_unregistered:.0%}")
        st.caption(f"{int(u.est_workers)} pekerja x 5% x UMK x 12 bulan. Upah usaha yang belum terdaftar tidak diketahui, "
                   "jadi UMK dipakai sebagai proksi konservatif. Waktu pemeriksaan "
                   f"{int(u.hours)} jam.")
    with st.expander("Kunci jawaban (khusus demo sintetis)"):
        st.caption("Tidak tersedia bagi petugas di dunia nyata. Tidak dipakai untuk menentukan peringkat.")
        t = UT.loc[nib]
        st.write({"wajib dan belum terdaftar": bool(t.is_violator), "jenis": t.kind, "pekerja sebenarnya": int(t.true_workers),
                  "Rp dapat dipulihkan (sebenarnya)": rp(t.recoverable_rp_true)})
    if branch in ("B", "C"):
        st.subheader("Hasil uji lima seed")
        us = load_csv(str(RESULTS / "unregistered_summary.csv"))
        us = us[us.branch == branch].set_index("ranking")
        runs = load_csv(str(RESULTS / "unregistered_runs.csv"))
        runs = runs[runs.branch == branch].pivot_table(index="seed", columns="ranking", values="rp_per_hour_juta")
        per_seed = runs.tepat_registration / runs.rules_only
        ratio = us.at["tepat_registration", "rp_h"] / us.at["rules_only", "rp_h"]
        with st.container(key="summary_unreg"):
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

# 3. Benchmark ---------------------------------------------------------------
if view == "Benchmark":
    rb = load_results()["branches"][branch]
    final = rb.get("final_ranking", "tepat_v2_1")
    best = rb["rankings"][final]
    baseline = rb["rankings"]["rules_only"]
    if branch == "A":
        st.warning("Cabang A dipakai untuk tuning. Angka utama harus berasal dari cabang B atau C.")
    st.warning("Hasil metode pada data sintetis. Kartu dan grafik di atas memakai seed bawaan; rentang lima seed ada di bawah. "
               "Bukan bukti dampak operasional BPJS atau estimasi nasional.")
    with st.container(key="summary"):
        a, b, c, d = st.columns(4)
        a.metric("Rp per jam pemeriksaan", jt(best["rp_per_hour"]))
        b.metric("Presisi pemeriksaan", f"{best['precision']:.0%}")
        c.metric("Perusahaan jujur ditandai", f"{angka(best['honest_flag_rate'] * 100)}%")
        d.metric("Dibanding aturan saja", kali(best["rp_per_hour"] / max(1, baseline["rp_per_hour"])),
                 help="Perbandingan Rp per jam pada seed bawaan, bukan dampak nasional.")
    st.caption(f"Cabang {branch}, seed bawaan: {ribu(rb['employers'])} perusahaan; {rb['violators']} pelanggar ditanam "
               f"({rb['prevalence']:.0%}); anggaran {rb['budget_hours']:.0f} jam, sekitar 10% total jam.")
    rows = []
    for k, v in rb["rankings"].items():
        rows.append(dict(Peringkat=RANK_LABEL.get(k, k), key=k, rp_per_jam=v["rp_per_hour"] / 1e6, presisi=v["precision"],
                         recall=v["recall"], jujur_ditandai=v["honest_flag_rate"], diperiksa=v["inspected"], rp_total=v["rp_recovered"] / 1e6))
    B = pd.DataFrame(rows)
    B["kelompok"] = np.where(B.key == final, "TEPAT utama", np.where(B.key.str.startswith("tepat"), "TEPAT lainnya", "Pembanding"))
    colors = alt.Color("kelompok:N", scale=alt.Scale(domain=["TEPAT utama", "TEPAT lainnya", "Pembanding"],
                                                     range=["#087f82", "#72aaa9", "#a6b7bd"]), legend=None)
    x1, x2 = st.columns(2, gap="large")
    with x1, st.container(border=True):
        st.subheader("Rupiah per jam")
        st.altair_chart(alt.Chart(B).mark_bar(cornerRadiusEnd=4, height=23).encode(
            x=alt.X("rp_per_jam:Q", title="Rp juta / jam pemeriksaan"), y=alt.Y("Peringkat:N", sort="-x", title=None), color=colors,
            tooltip=["Peringkat", alt.Tooltip("rp_per_jam:Q", format=".2f"), "diperiksa"]).properties(height=275).configure(**CHART_CONFIG),
            theme=None, width="stretch")
    with x2, st.container(border=True):
        st.subheader("Presisi pemeriksaan")
        st.altair_chart(alt.Chart(B).mark_bar(cornerRadiusEnd=4, height=23).encode(
            x=alt.X("presisi:Q", title="Bagian yang diperiksa memang melanggar", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Peringkat:N", sort="-x", title=None), color=colors,
            tooltip=["Peringkat", alt.Tooltip("presisi:Q", format=".0%")]).properties(height=275).configure(**CHART_CONFIG),
            theme=None, width="stretch")
    st.subheader("Hasil lengkap")
    table = B.drop(columns=["key", "kelompok"]).rename(columns={"Peringkat": "Metode", "rp_per_jam": "Rp jt/jam",
        "presisi": "Presisi", "recall": "Recall", "jujur_ditandai": "Jujur ditandai", "diperiksa": "Diperiksa", "rp_total": "Rp jt total"})
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
    rb = load_results()["branches"][branch]
    arm = rb["rankings"].get(f"{rb.get('final_ranking', 'tepat_v2_1')}_with_random_arm", {})
    with st.container(key="summary"):
        a, b, c = st.columns(3)
        a.metric("Anggaran jalur acak", "8%")
        b.metric("Perusahaan sampel acak", f"{arm.get('random_arm_size', 0)}")
        c.metric("Pelanggaran di sampel acak", f"{arm.get('random_arm_violation_rate', 0):.0%}")
    st.info("Jalur acak berstrata menurut sektor dan skala memberi kesempatan pemeriksaan di luar daftar berisiko tinggi. "
            "Sampel kecil ini membantu memantau bias seleksi; bukan angka prevalensi nasional.")
    fr = pd.DataFrame(rb.get("fairness", []))
    if len(fr):
        for grp, title in (("sector", "Sektor"), ("band_obs", "Skala usaha")):
            with st.container(border=True):
                st.subheader(f"Pola pemeriksaan menurut {title.lower()}")
                g = fr[fr.group == grp].copy().rename(columns={"value": "grp"})
                if grp == "sector":
                    g["grp"] = g.grp.map(SEKTOR)
                m = g.melt(id_vars=["grp"], value_vars=["violation_rate", "pick_rate", "honest_flag_rate"], value_name="rate")
                m["variable"] = m.variable.map({"violation_rate": "Pelanggaran sebenarnya", "pick_rate": "Dipilih TEPAT", "honest_flag_rate": "Jujur ikut terpilih"})
                st.altair_chart(alt.Chart(m).mark_bar(cornerRadiusEnd=3).encode(
                    x=alt.X("grp:N", title=None, axis=alt.Axis(labelAngle=-25, labelLimit=145)), xOffset="variable:N",
                    y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title=None),
                    color=alt.Color("variable:N", scale=alt.Scale(domain=["Pelanggaran sebenarnya", "Dipilih TEPAT", "Jujur ikut terpilih"],
                                                                range=["#98aeb5", "#087f82", "#d48b4a"]), legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["grp", "variable", alt.Tooltip("rate:Q", format=".1%")]).properties(height=260).configure(**CHART_CONFIG),
                    theme=None, width="stretch")
    st.caption("Skala mikro & kecil: upah di bawah UMK bukan pelanggaran (Perpres 59/2024 Ps. 32(4), E05c). Aturan itu hanya berlaku "
               "untuk usaha menengah & besar. Label pelanggaran sebenarnya hanya digunakan pada evaluasi sintetis ini.")

if view == "Cek kepesertaan saya":
    st.info("Untuk pekerja: apakah pemberi kerja sudah mendaftarkan dan membayar iuran BPJS Kesehatan Anda? "
               "Di dunia nyata fitur ini berada di Mobile JKN; di sini memakai data sintetis.")
    if demo_mode:
        n_detail = demo_worker_employers(branch)
        st.caption(f"Versi daring ini memuat data pekerja dari {n_detail} perusahaan contoh, bukan {ribu(len(reg))}. "
                   "Hasil untuk setiap NIK yang dimuat tetap sama dengan data lengkap. Halaman lain memakai semua perusahaan.")
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
    pick = st.radio("Contoh NIK sintetis", list(demos), horizontal=False, key="worker_scenario")
    nik = st.text_input("NIK (16 digit)", value=demos.get(pick, ""), max_chars=16)
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
            if st.button("Kirim laporan demo", key=f"lapor_{nik}_{i}", type="primary"):
                tiket = add_tip(nik, emp, jenis)
                st.success(f"Laporan demo tersimpan. Nomor tiket **{tiket}**. NIK mentah tidak disimpan dalam tiket; hanya hash. "
                           "Laporan muncul di halaman kasus perusahaan untuk demo petugas. Tidak dikirim ke BPJS.")
        if len(my_ks) == 0 and len(my_tk) == 0 and len(my_dep) == 0:
            st.info("NIK tidak ditemukan" + (" dalam data pekerja versi daring" if demo_mode else "")
                    + ". Pilih skenario demo untuk mencoba data yang tersedia.")
    st.caption("Dalam proposal: notifikasi proaktif lewat Mobile JKN dan alur triase laporan (slide), bukan bagian prototipe.")

st.markdown('<footer class="app-footer"><span>TEPAT / Tim Johannes Kepler / Healthkathon 2026</span>'
            '<span>Data sintetis. Keputusan pemeriksaan tetap pada petugas.</span></footer>', unsafe_allow_html=True)
