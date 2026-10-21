"""Evidence-grounded drafting (src/tepat/drafting.py) and the 'Draf dokumen' panel in the case view."""
import io
import json
import re
import threading
import time
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tepat import drafting as Dr
from tepat import scorer

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
DEMO = ROOT / "demo"
ISSUED = date(2026, 10, 21)
GLYPHS = ("\u2014", "\u2013", "\u00b7", "\u2026")
SEKTOR = {"C": "Manufaktur", "F": "Konstruksi", "G": "Perdagangan", "H": "Logistik", "I": "Akomodasi & Makan",
          "J": "Informasi & Kom.", "K": "Keuangan", "Q": "Kesehatan", "A": "Pertanian"}


def load(branch: str = "B"):
    ranked = pd.read_parquet(DEMO / branch / "ranked.parquet")
    F = pd.read_parquet(DEMO / branch / "features.parquet")
    reg = pd.read_parquet(DEMO / branch / "observed" / "employer_registry.parquet").set_index("employer_id")
    return ranked, F, reg


RANKED, F, REG = load("B")


def ev_for(eid: str) -> Dr.Evidence:
    return Dr.evidence(eid, RANKED.loc[eid], F.loc[eid], REG.loc[eid], SEKTOR[REG.at[eid, "sector"]])


def _top_with_unregistered_workers() -> str:
    """Highest-ranked inspection case whose letter cites UU 24/2011 Ps. 15 and not Ps. 14 (used by edits below)."""
    for eid in RANKED.index[RANKED.queue == "inspeksi"]:
        kinds = [Dr.kind_of(t) for _, t, _ in scorer.reasons(F.loc[eid])]
        if "unreg" in kinds and "piggy" not in kinds:
            return eid
    raise AssertionError("demo slice has no case with unregistered workers")


TOP = _top_with_unregistered_workers()
CICILAN_ID = RANKED.index[RANKED.queue == "cicilan"][0]


# --------------------------------------------------------------------------- templates

def test_inspection_brief_carries_the_case_evidence():
    ev = ev_for(TOP)
    d = Dr.draft(ev, Dr.BRIEF, issued=ISSUED)
    assert d.text.splitlines()[0] == Dr.LABEL
    assert REG.at[TOP, "name"] in d.text and TOP in d.text
    assert f"Peringkat {int(RANKED.at[TOP, 'rank'])}" in d.text
    assert f"P(pelanggaran): {RANKED.at[TOP, 'p_violation']:.0%}" in d.text
    assert Dr.rp(RANKED.at[TOP, "est_rp"]) in d.text and Dr.rp(RANKED.at[TOP, "rp_high"]) in d.text
    for _, text, basis in scorer.reasons(F.loc[TOP])[:5]:
        assert text in d.text and f"Dasar: {basis}" in d.text
    for doc in scorer.brief(F.loc[TOP]):
        assert f"- {doc}" in d.text
    assert "Pertanyaan saat kunjungan" in d.text and ev.questions and ev.questions[0] in d.text
    assert RANKED.at[TOP, "action"] in d.text


def test_compliance_letter_is_a_self_correction_offer_with_a_deadline():
    ev = ev_for(TOP)
    d = Dr.draft(ev, Dr.COMPLIANCE, deadline_days=14, issued=ISSUED)
    assert d.text.splitlines()[0] == Dr.LABEL
    assert "14 hari kalender" in d.text and "paling lambat 4 November 2026" in d.text
    assert "belum tentu berarti pelanggaran" in d.text and "koreksi mandiri" in d.text
    assert "Placeholder sintetis" in d.text and "[NAMA PETUGAS]" in d.text
    for text, basis in ev.letter_points:
        assert text in d.text and f"Dasar: {basis}" in d.text
    longer = Dr.draft(ev, Dr.COMPLIANCE, deadline_days=21, issued=ISSUED)
    assert "21 hari kalender" in longer.text and "11 November 2026" in longer.text


def test_cicilan_letter_states_arrears_and_the_instalment_path():
    ev = ev_for(CICILAN_ID)
    assert Dr.available_doc_types(ev) == [Dr.BRIEF, Dr.CICILAN]
    d = Dr.draft(ev, Dr.CICILAN, issued=ISSUED)
    assert d.text.splitlines()[0] == Dr.LABEL
    assert "program pembayaran bertahap" in d.text and "bukan surat teguran" in d.text
    assert "Program REHAB (Rencana Pembayaran Bertahap)" in d.text
    assert f"terlambat pada {ev.ks_late_months} dari 12 bulan" in d.text
    assert Dr.rp(F.at[CICILAN_ID, "rp_arrears"] * 12) in d.text
    assert "P(pelanggaran)" not in d.text and "Peringkat" not in d.text


def test_cicilan_letter_is_refused_outside_the_cicilan_queue():
    ev = ev_for(TOP)
    assert ev.queue == "inspeksi" and Dr.CICILAN not in Dr.available_doc_types(ev)
    with pytest.raises(Dr.DraftNotAvailable):
        Dr.draft(ev, Dr.CICILAN, issued=ISSUED)


def test_employer_letter_leaves_out_model_probability_rank_and_rp():
    ids = list(RANKED.index[RANKED.queue == "inspeksi"][:60])
    checked = 0
    for eid in ids:
        ev = ev_for(eid)
        if Dr.COMPLIANCE not in Dr.available_doc_types(ev):
            continue
        text = Dr.draft(ev, Dr.COMPLIANCE, issued=ISSUED).text
        checked += 1
        assert "Rp" not in text, eid
        low = text.lower()
        assert "peringkat" not in low and "p(pelanggaran)" not in low and "probabilitas" not in low, eid
        assert ev.tier not in text, eid
        # The only percentage allowed is the 1% employee share in a document request.
        for line in text.splitlines():
            if "%" in line:
                assert re.findall(r"\d+%", line) == ["1%"] and "iuran 1% dari gaji" in line, (eid, line)
    assert checked >= 40


def test_drafting_never_reads_the_answer_key():
    for path in (ROOT / "src" / "tepat" / "drafting.py", ROOT / "app" / "drafting_panel.py"):
        src = path.read_text(encoding="utf-8")
        assert "truth" not in src and "generator" not in src and "bench" not in src, path.name


def test_every_scorer_reason_has_a_drafting_kind():
    for branch in ("A", "B", "C"):
        _, Fb, _ = load(branch)
        for eid in Fb.index:
            for _, text, _ in scorer.reasons(Fb.loc[eid]):
                assert Dr.kind_of(text) is not None, text


def test_no_banned_glyphs_in_drafts_or_new_sources():
    for path in (ROOT / "src" / "tepat" / "drafting.py", ROOT / "app" / "drafting_panel.py", Path(__file__)):
        src = path.read_text(encoding="utf-8")
        for g in GLYPHS:
            assert g not in src, f"{path.name} uses {g!r}"
    for eid in RANKED.index:
        ev = ev_for(eid)
        for t in Dr.available_doc_types(ev):
            d = Dr.draft(ev, t, issued=ISSUED)
            assert not any(g in d.text for g in GLYPHS), (eid, t)
            assert d.text.startswith(Dr.LABEL)
            assert Dr.check(d.text, d, ev).ok, (eid, t, Dr.check(d.text, d, ev).problems)


# --------------------------------------------------------------------------- checker and LLM rewrite

def letter():
    ev = ev_for(TOP)
    return ev, Dr.draft(ev, Dr.COMPLIANCE, issued=ISSUED)


def faithful(text: str) -> str:
    """Wording changes only, including a long-form citation of a basis already in the template."""
    out = (text.replace("Kami memberikan kesempatan kepada Bapak/Ibu untuk memeriksa",
                        "Kami memberi Bapak/Ibu kesempatan untuk memeriksa")
               .replace("Perbedaan data belum tentu berarti pelanggaran.", "Perbedaan ini belum tentu menunjukkan pelanggaran.")
               .replace("Dasar: UU 24/2011 Ps. 15;", "Dasar: Pasal 15 Undang-Undang Nomor 24 Tahun 2011;"))
    assert out != text
    return "```text\n" + out + "```"


def test_faithful_rewrite_passes_the_fact_check():
    ev, d = letter()
    res = Dr.rewrite(d, ev, client=lambda messages: faithful(d.text))
    assert res.source == "llm", res.problems
    assert "Kami memberi Bapak/Ibu kesempatan" in res.text and not res.text.startswith("```")


def test_prompt_carries_the_template_and_the_facts_only():
    ev, d = letter()
    seen = {}
    Dr.rewrite(d, ev, client=lambda m: seen.setdefault("m", m) and d.text)
    user = seen["m"][1]["content"]
    assert d.text in user and "FAKTA BUKTI" in user
    assert Dr.rp(ev.est_rp) not in user and f"{ev.p_violation:.0%}" not in user.replace("1%", "")


@pytest.mark.parametrize("edit, expected", [
    (lambda t: t.replace("Dengan hormat,", "Dengan hormat,\nTunggakan mencapai Rp987.654.321."), "987654321"),
    (lambda t: t.replace("Dasar: UU 24/2011 Ps. 15;", "Dasar: UU 24/2011 Ps. 14;"), "dasar hukum"),
    (lambda t: t.replace("Dasar: UU 24/2011 Ps. 15;", "Dasar: UU 24/2011 Ps. 15; Permenkes 3/2023;"), "permenkes 3/2023"),
    (lambda t: t + "\nSalinan untuk PT Sentosa Abadi.", "nama perusahaan tidak dikenal: PT Sentosa Abadi"),
    (lambda t: t.replace("[NAMA PETUGAS]", "Budi Santoso"), "Budi"),
    (lambda t: t.replace("paling lambat 4 November 2026", "paling lambat 14 Oktober 2026"), "tanggal"),
    (lambda t: t.replace("Dengan hormat,", "Dengan hormat \u2014"), "U+2014"),
    (lambda t: t.replace(Dr.LABEL, "Surat resmi"), "label DRAF"),
    (lambda t: t.replace("Dengan hormat,", "Dengan hormat, perusahaan Anda di peringkat atas."), "data internal"),
])
def test_unfaithful_rewrite_falls_back_to_the_template(edit, expected):
    ev, d = letter()
    res = Dr.rewrite(d, ev, client=lambda messages: edit(d.text))
    assert res.source == "template" and res.text == d.text
    assert any(expected in p for p in res.problems), res.problems


def test_timeout_and_errors_fall_back_to_the_template():
    ev, d = letter()

    def slow(messages):
        raise TimeoutError("timed out")

    res = Dr.rewrite(d, ev, client=slow)
    assert res.source == "template" and res.text == d.text and "batas waktu" in res.problems[0]
    res = Dr.rewrite(d, ev, client=lambda m: {}["choices"])
    assert res.source == "template" and "KeyError" in res.problems[0]
    res = Dr.rewrite(d, ev)
    assert res.source == "template" and not res.problems and "tidak dikonfigurasi" in res.note


class _Server:
    """A local OpenAI-compatible endpoint, so the urllib client is exercised without leaving the machine."""

    def __init__(self, reply: str, delay: float = 0.0):
        seen = self.seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"], seen["auth"] = self.path, self.headers.get("Authorization")
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                time.sleep(delay)
                data = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:
                    pass

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_http_client_speaks_openai_compatible_chat_completions():
    ev, d = letter()
    server = _Server(faithful(d.text))
    try:
        cfg = Dr.LLMConfig("sk-test-not-real", server.url, "demo-model", timeout_s=10)
        res = Dr.rewrite(d, ev, cfg)
    finally:
        server.close()
    assert res.source == "llm", res.problems
    assert server.seen["path"] == "/v1/chat/completions"
    assert server.seen["auth"] == "Bearer sk-test-not-real"
    assert server.seen["body"]["model"] == "demo-model"
    assert "sk-test-not-real" not in repr(cfg) and "sk-test-not-real" not in json.dumps(server.seen["body"])


def test_http_timeout_falls_back_to_the_template():
    ev, d = letter()
    server = _Server(d.text, delay=3.0)
    try:
        res = Dr.rewrite(d, ev, Dr.LLMConfig("k", server.url, "m", timeout_s=0.5))
    finally:
        server.close()
    assert res.source == "template" and res.text == d.text
    assert res.problems and "batas waktu" in res.problems[0]


def test_llm_config_needs_all_three_values():
    full = {"TEPAT_LLM_API_KEY": "k", "TEPAT_LLM_BASE_URL": "http://x/v1", "TEPAT_LLM_MODEL": "m"}
    assert Dr.LLMConfig.from_values(full) is not None
    for missing in full:
        assert Dr.LLMConfig.from_values({**full, missing: ""}) is None


# --------------------------------------------------------------------------- DOCX

def test_docx_round_trip_keeps_heading_and_reasons():
    from docx import Document

    ev = ev_for(TOP)
    d = Dr.draft(ev, Dr.BRIEF, issued=ISSUED)
    doc = Document(io.BytesIO(Dr.to_docx(d.text, d.title, d.headings)))
    paras = [(p.style.name, p.text) for p in doc.paragraphs]
    assert ("Heading 1", d.title) in paras
    assert ("Heading 2", "Alasan utama") in paras
    first_reason = scorer.reasons(F.loc[TOP])[0][1]
    assert any(first_reason in text for _, text in paras)
    assert paras[0][1] == Dr.LABEL


# --------------------------------------------------------------------------- app

def test_panel_sits_before_the_answer_key_in_the_case_view():
    src = APP.read_text(encoding="utf-8")
    case = src[src.index('if view == "Kasus perusahaan":'):src.index('if view == "Usaha belum terdaftar":')]
    assert case.index("drafting_panel.render(") < case.index("Kunci jawaban")
    assert "def on_draft_approved(employer_id: str, doc_type: str, text: str)" in case
    assert "store.record_draft_approval(" in case  # approvals go into the case history


def test_case_view_drafts_and_approves_without_sending(monkeypatch):
    monkeypatch.setenv("TEPAT_DEMO", "1")
    for k in ("TEPAT_LLM_API_KEY", "TEPAT_LLM_BASE_URL", "TEPAT_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.sidebar.radio(key="view").set_value("Kasus perusahaan").run()
    assert not app.exception
    assert app.radio(key="draft_type").options == [Dr.DOC_LABELS[Dr.BRIEF], Dr.DOC_LABELS[Dr.COMPLIANCE]]
    app.radio(key="draft_type").set_value(Dr.COMPLIANCE).run()
    app.button(key="draft_generate").click().run()
    assert not app.exception
    text = app.code[0].value
    assert text.startswith(Dr.LABEL) and "Rp" not in text
    app.button(key="draft_approve").click().run()
    assert not app.exception
    eid = app.selectbox(key="case_employer").value
    stored = app.session_state["approved_drafts"][f"{eid}:{Dr.COMPLIANCE}"]
    assert stored["text"].strip() == text.strip() and stored["doc_type"] == Dr.COMPLIANCE
    assert any("Tidak ada yang dikirim" in s.value for s in app.success)
    # A cicilan-queue case offers the instalment letter.
    app.selectbox(key="case_employer").set_value(CICILAN_ID).run()
    app.radio(key="draft_type").set_value(Dr.CICILAN).run()
    app.button(key="draft_generate").click().run()
    assert not app.exception
    assert "program pembayaran bertahap" in app.code[0].value
