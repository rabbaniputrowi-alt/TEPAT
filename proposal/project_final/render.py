"""Render the deck to PDF with PowerPoint (COM), then to per-slide PNGs for inspection.

Run: uv run --with pywin32 --with pymupdf python proposal/project_final/render.py [PPTX]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
import win32com.client

HERE = Path(__file__).resolve().parent
pptx = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out" / "TEPAT Project Final.pptx"
pdf = pptx.with_suffix(".pdf")
prev = HERE / "out" / "preview"
prev.mkdir(parents=True, exist_ok=True)

app = win32com.client.DispatchEx("PowerPoint.Application")
try:
    pres = app.Presentations.Open(str(pptx), ReadOnly=True, Untitled=False, WithWindow=False)
    pres.SaveAs(str(pdf), 32)  # ppSaveAsPDF
    pres.Close()
finally:
    app.Quit()

doc = pymupdf.open(pdf)
fonts = sorted({f[3] for p in doc for f in p.get_fonts()})
for i, page in enumerate(doc, 1):
    page.get_pixmap(dpi=144).save(prev / f"s{i:02d}.png")
banned = {"\u2014", "\u2013", "\u00b7", "\u2026"}
txt = "".join(p.get_text() for p in doc)
print({"pdf": str(pdf), "pages": len(doc), "bytes": pdf.stat().st_size, "fonts": fonts,
       "banned_in_pdf": sum(txt.count(b) for b in banned), "size_pt": tuple(round(v, 1) for v in doc[0].rect[2:])})
