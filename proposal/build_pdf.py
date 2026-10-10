"""Build the proposal PDF from TEPAT Proposal.html and check it.

Run: uv run --with playwright --with pypdf python proposal/build_pdf.py <out.pdf>
Checks: every slide's content fits its frame (no clipped overflow), 20 pages, size under 10 MB,
no em dash, en dash, middle dot or ellipsis glyphs in the PDF text. Writes PNG previews to proposal/preview/.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "TEPAT Proposal.html"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "TEPAT Proposal.pdf"
PREVIEW = ROOT / "preview"
PREVIEW.mkdir(exist_ok=True)

CHECK = """() => [...document.querySelectorAll('.slide')].map(s => {
  const r = s.getBoundingClientRect();
  const foot = s.querySelector('.foot');
  const limit = foot ? foot.getBoundingClientRect().top : r.bottom - 40;
  let worst = 0, culprit = '';
  s.querySelectorAll('.body *, h2, .cover *').forEach(e => {
    if (foot && foot.contains(e)) return;
    const b = e.getBoundingClientRect();
    if (b.height === 0) return;
    const over = b.bottom - limit;
    if (over > worst) { worst = over; culprit = (e.tagName + ' ' + (e.textContent || '').trim().slice(0, 60)); }
    const side = b.right - (r.right - 100);
    if (side > worst) { worst = side; culprit = 'RIGHT ' + e.tagName + ' ' + (e.textContent || '').trim().slice(0, 60); }
  });
  const fo = foot ? (foot.getBoundingClientRect().bottom - r.bottom) : 0;
  return {id: s.id, overflow_px: Math.round(worst), culprit, foot_past_edge_px: Math.round(fo)};
})"""

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe", headless=True)
    page = browser.new_page(viewport={"width": 1968, "height": 1100}, device_scale_factor=1)
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("requestfailed", lambda r: errors.append("requestfailed " + r.url))
    page.goto(HTML.as_uri(), wait_until="load")
    page.evaluate("document.fonts.ready")
    page.evaluate("document.documentElement.style.setProperty('--z', '1')")
    font_ok = page.evaluate("document.fonts.check('700 30px \"Public Sans\"')")
    report = page.evaluate(CHECK)
    for i, s in enumerate(page.locator(".slide").all()):
        s.screenshot(path=str(PREVIEW / f"s{i + 1:02d}.png"))
    page.emulate_media(media="print")
    page.pdf(path=str(OUT), width="1920px", height="1080px", print_background=True, prefer_css_page_size=True)
    browser.close()

reader = PdfReader(str(OUT))
text = "\n".join(pg.extract_text() or "" for pg in reader.pages)
bad = re.findall("[\u2014\u2013\u00b7\u2026]", text)
size_mb = OUT.stat().st_size / 1e6
problems = [r for r in report if r["overflow_px"] > 0 or r["foot_past_edge_px"] > 0]
print(json.dumps({
    "pdf": str(OUT), "pages": len(reader.pages), "size_mb": round(size_mb, 2), "font_loaded": font_ok,
    "banned_glyphs": len(bad), "page_errors": errors, "overflowing_slides": problems,
}, indent=2, ensure_ascii=False))
