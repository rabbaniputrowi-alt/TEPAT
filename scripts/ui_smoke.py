"""Browser smoke/visual capture. Run with uv run --with playwright python scripts/ui_smoke.py.
Uses an isolated headless Chrome profile, not the user's personal browser.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VIEWS = ["Daftar prioritas", "Kasus perusahaan", "Tindak lanjut", "Usaha belum terdaftar", "Benchmark", "Keadilan & jalur acak",
         "Cek kepesertaan saya"]
SCROLL_VIEWS = ("Tindak lanjut", "Usaha belum terdaftar", "Benchmark")   # long views: also capture each screen below the fold
parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8501")
parser.add_argument("--out", default=str(ROOT / ".impeccable" / "review"))
args = parser.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
records = []
errors = []

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe", headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1, reduced_motion="reduce")
    page = context.new_page()
    page.on("pageerror", lambda err: errors.append(str(err)))
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.goto(args.url, wait_until="domcontentloaded")
    # First paint on a cold server includes loading and scoring the synthetic branch.
    page.get_by_role("button", name="Buka kasus", exact=True).wait_for(timeout=300000)
    if page.get_by_role("button", name="Don't show again", exact=True).is_visible():
        page.get_by_role("button", name="Don't show again", exact=True).click()

    def settle(title: str):
        # A full rerun takes 15-21 s on this laptop on battery; 30 s left too little margin after a reload.
        page.get_by_role("heading", name=title, exact=True).wait_for(timeout=90000)
        page.wait_for_function("document.querySelectorAll('[data-testid=stSkeleton]').length === 0", timeout=90000)
        page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').wait_for(timeout=90000)
        page.evaluate("document.fonts.ready")
        page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
        assert page.locator('[data-testid="stException"]').count() == 0

    for n, view in enumerate(VIEWS):
        page.locator('[data-testid="stSidebar"]').get_by_text(view, exact=True).click()
        settle(view)
        if view == "Daftar prioritas":
            page.get_by_role("button", name="Buka kasus", exact=True).wait_for()
            page.locator('[data-testid="stDataFrame"] canvas').first.wait_for()
        if view in ("Kasus perusahaan", "Benchmark", "Keadilan & jalur acak"):
            page.locator('[data-testid="stVegaLiteChart"]').first.wait_for(timeout=30000)
            page.wait_for_function("document.querySelectorAll('[data-testid=stVegaLiteChart] canvas, [data-testid=stVegaLiteChart] svg').length > 0", timeout=30000)
        page.locator('[data-testid="stMain"]').evaluate("e => e.scrollTop = 0")
        path = out / f"desktop-{n + 1}.png"
        page.screenshot(path=str(path))
        records.append({"viewport": "desktop", "view": view, "screenshot": str(path),
                        "metrics": page.locator('[data-testid="stMetricValue"]').all_text_contents()})
        if view in SCROLL_VIEWS:
            main = page.locator('[data-testid="stMain"]')
            height = main.evaluate("e => e.scrollHeight")
            for k, top in enumerate(range(900, height, 900), start=1):
                main.evaluate(f"e => e.scrollTop = {top}")
                page.wait_for_timeout(400)
                page.screenshot(path=str(out / f"desktop-{n + 1}-s{k}.png"))
            main.evaluate("e => e.scrollTop = 0")
        (out / "browser-results.json").write_text(json.dumps({"records": records, "errors": errors}, indent=2, ensure_ascii=False), encoding="utf-8")

    # Exercise the real queue-to-case path.
    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.get_by_text("Daftar prioritas", exact=True).click()
    settle("Daftar prioritas")
    name = page.locator('.case-preview-name').inner_text()
    page.get_by_role("button", name="Buka kasus", exact=True).click()
    settle("Kasus perusahaan")
    page.get_by_role("heading", name=name, exact=True).wait_for(timeout=30000)
    print("queue_to_case", name)

    # Mobile layout and navigation: collapsed sidebar, no page-wide overflow.
    page.set_viewport_size({"width": 390, "height": 844})
    page.reload(wait_until="domcontentloaded")
    page.get_by_role("heading", name="Daftar prioritas", exact=True).wait_for(timeout=30000)
    settle("Daftar prioritas")
    labels = page.get_by_test_id("stMetricLabel").locator("p").evaluate_all(
        "xs => xs.map(e => ({text: e.textContent, whiteSpace: getComputedStyle(e).whiteSpace}))")
    assert all(label["whiteSpace"] != "nowrap" for label in labels), labels
    page.screenshot(path=str(out / "mobile-queue.png"))
    overflow = page.evaluate("({width:innerWidth, scroll:document.documentElement.scrollWidth, main:document.querySelector('[data-testid=stMain]').scrollWidth})")
    records.append({"viewport": "mobile", "view": "Daftar prioritas", "screenshot": str(out / "mobile-queue.png"), "overflow": overflow})
    toggle = page.get_by_test_id("stExpandSidebarButton")
    assert toggle.is_visible(), "Mobile users need a visible navigation control."
    toggle.click()
    sidebar.get_by_text("Cek kepesertaan saya", exact=True).click()
    settle("Cek kepesertaan saya")
    collapse = sidebar.locator('[data-testid="stSidebarCollapseButton"] button')
    if collapse.count() and collapse.is_visible():
        collapse.click()
    page.locator('[data-testid="stMain"]').evaluate("e => e.scrollTop = 0")
    page.screenshot(path=str(out / "mobile-worker.png"))
    records.append({"viewport": "mobile", "view": "Cek kepesertaan saya", "screenshot": str(out / "mobile-worker.png")})
    print("mobile_overflow", overflow)
    print("computed_styles", page.evaluate("({font:getComputedStyle(document.querySelector('h1')).fontFamily, background:getComputedStyle(document.querySelector('[data-testid=stApp]')).backgroundColor})"))
    data = {"records": records, "errors": errors, "desktop_views": len([r for r in records if r['viewport'] == 'desktop'])}
    (out / "browser-results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    assert data["desktop_views"] == len(VIEWS)
    assert not errors, errors
    assert overflow["scroll"] <= overflow["width"]
    assert overflow["main"] <= overflow["width"]
    print(json.dumps(data, indent=2, ensure_ascii=False))
    browser.close()
