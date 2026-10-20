"""'Draf dokumen' panel for the case view (feature/drafting).

Drafts come from observed evidence only (scorer output, features, registry); the answer key is never
passed in. The optional LLM rewrite is configured with TEPAT_LLM_API_KEY, TEPAT_LLM_BASE_URL and
TEPAT_LLM_MODEL from the environment or st.secrets. Nothing is sent to the employer: approval only marks
the draft in session state through the `on_approved` callback.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable

import pandas as pd
import streamlit as st

from tepat import drafting

LLM_KEYS = ("TEPAT_LLM_API_KEY", "TEPAT_LLM_BASE_URL", "TEPAT_LLM_MODEL")


def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:  # no secrets.toml: Streamlit raises instead of returning a default
        return ""


def llm_config() -> drafting.LLMConfig | None:
    """Environment first, then st.secrets. The key is never displayed or logged."""
    return drafting.LLMConfig.from_values({k: os.environ.get(k) or _secret(k) for k in LLM_KEYS})


def _markdown(text: str, title: str, headings: tuple[str, ...]) -> str:
    """Plain-text draft to safe Markdown for the preview: escape Markdown syntax, mark headings."""
    out = []
    for line in text.splitlines():
        s = re.sub(r"([\\`*_{}\[\]<>#|~$])", r"\\\1", line.strip())
        if not s:
            out.append("")
        elif line.strip() == title or line.strip() in headings:
            out += [f"**{s}**", ""]
        elif line.startswith("- "):
            out.append("- " + s[2:])
        elif line.startswith("   "):
            out.append("   " + s + "  ")
        else:
            out.append(s + "  ")
    return "\n".join(out)


def render(employer_id: str, ranked_row: pd.Series, features_row: pd.Series, registry_row: pd.Series,
           sector_label: str, on_approved: Callable[[str, str, str], None]) -> None:
    ev = drafting.evidence(employer_id, ranked_row, features_row, registry_row, sector_label)
    types = drafting.available_doc_types(ev)
    with st.container(border=True, key="draft_panel"):
        st.subheader("Draf dokumen")
        st.caption("Disusun dari bukti yang teramati: sinyal skor, data kepesertaan, dan registri. Kunci jawaban tidak "
                   "dipakai. Petugas meninjau dan menyetujui; tidak ada yang dikirim.")
        if st.session_state.get("draft_type") not in types:   # e.g. cicilan picked, then a non-cicilan case
            st.session_state["draft_type"] = types[0]
        doc_type = st.radio("Jenis dokumen", types, key="draft_type", horizontal=True,
                            format_func=lambda t: drafting.DOC_LABELS[t])
        if drafting.CICILAN not in types:
            st.caption("Surat penawaran cicilan hanya tersedia untuk perusahaan di antrean cicilan.")
        days = drafting.DEFAULT_DEADLINE_DAYS
        if doc_type != drafting.BRIEF:
            days = int(st.number_input("Tenggat tanggapan (hari kalender)", min_value=7, max_value=60,
                                       value=drafting.DEFAULT_DEADLINE_DAYS, step=1, key="draft_deadline"))
        cfg = llm_config()
        use_llm = st.toggle("Perhalus bahasa dengan LLM", value=False, disabled=cfg is None, key="draft_llm")
        if cfg is None:
            st.caption("LLM belum dikonfigurasi (TEPAT_LLM_API_KEY, TEPAT_LLM_BASE_URL, TEPAT_LLM_MODEL). "
                       "Template deterministik dipakai.")
        store = st.session_state.setdefault("draft_results", {})
        slot = f"{employer_id}:{doc_type}"
        if st.button("Buat draf", key="draft_generate"):
            d = drafting.draft(ev, doc_type, deadline_days=days)
            if use_llm and cfg is not None:
                with st.spinner("Memperhalus bahasa dan memeriksa fakta"):
                    res = drafting.rewrite(d, ev, cfg)
            else:
                res = drafting.Result(d.text, "template", [], "")
            store[slot] = {"draft": d, "result": res, "days": days}
            st.session_state.setdefault("draft_approved", set()).discard(slot)   # a new text needs a new approval
        current = store.get(slot)
        if current is None:
            st.caption("Pilih jenis dokumen, lalu tekan Buat draf.")
            return
        d, res = current["draft"], current["result"]
        if doc_type != drafting.BRIEF and current["days"] != days:
            st.caption("Tenggat berubah. Tekan Buat draf untuk memperbarui teks.")
        if res.source == "llm":
            st.caption(f"Sumber teks: LLM. {res.note}")
        else:
            st.caption(f"Sumber teks: template deterministik. {res.note}".strip())
        if res.problems:
            st.warning("Hasil LLM tidak dipakai. Alasan: " + "; ".join(res.problems))
        preview, plain = st.tabs(["Pratinjau", "Teks siap salin"])
        with preview, st.container(border=True, key="draft_preview"):
            st.markdown(_markdown(res.text, d.title, d.headings))
        with plain:
            st.code(res.text, language=None, wrap_lines=True)
        st.download_button("Unduh DOCX", data=drafting.to_docx(res.text, d.title, d.headings),
                           file_name=f"draf_{doc_type}_{employer_id}.docx", key="draft_download",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        approved = st.session_state.setdefault("draft_approved", set())
        if st.button("Setujui draf", key="draft_approve", type="primary"):
            approved.add(slot)
            on_approved(employer_id, doc_type, res.text)
        if slot in approved:
            st.success("Draf disetujui dan ditandai di sesi ini. Tidak ada yang dikirim ke pemberi kerja atau ke "
                       "sistem BPJS mana pun.")
