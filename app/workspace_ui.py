"""Small visual primitives for the TEPAT workspace, using native Streamlit widgets."""
from __future__ import annotations

from html import escape
from typing import Sequence

import streamlit as st


def kpis(items: Sequence[tuple[str, str, str]], *, key: str = "workspace_kpis") -> None:
    """Four source-labelled metrics; native widgets remain visible to AppTest and assistive tech."""
    with st.container(key=key):
        for column, (label, value, note) in zip(st.columns(len(items), gap="small"), items):
            with column:
                st.metric(label, value)
                if note:
                    st.caption(note)


def row_badge(text: str, tone: str = "quiet") -> str:
    return f'<span class="ws-badge ws-badge-{tone}">{escape(str(text))}</span>'


def bar_rows(rows: Sequence[tuple[str, float, str, str]], *, maximum: float) -> str:
    """Accessible bar rows. The numeric value stays visible as text and drives the bar width."""
    return '<div class="ws-bars">' + "".join(
        f'<div class="ws-bar-row"><span>{escape(label)}</span>'
        f'<span class="ws-bar-track"><span class="ws-bar-fill ws-bar-{escape(tone)}" '
        f'style="width:{max(0.0, min(100.0, value / maximum * 100)):.2f}%"></span></span>'
        f'<strong>{escape(display)}</strong></div>'
        for label, value, display, tone in rows
    ) + '</div>'


def ordered_reasons(reasons: Sequence[tuple[str, str]]) -> str:
    return "".join(
        '<div class="ws-reason"><span class="ws-reason-index">'
        f'{index}</span><div><strong>{escape(str(text))}</strong>'
        f'<small>Dasar: {escape(str(basis))}</small></div></div>'
        for index, (text, basis) in enumerate(reasons, 1)
    )
