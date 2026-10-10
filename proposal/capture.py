"""Capture app screens for the proposal deck from a running local server.

Run: uv run --with playwright python proposal/capture.py [--url http://localhost:8501]
Uses an isolated headless Chrome profile. Branch B (held-out test), full local run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets"
CASE = "PT Baru Sejahtera (peringkat 26)"

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8501")
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
errors: list[str] = []

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe", headless=True)
    ctx = browser.new_context(viewport={"width": 1600, "height": 1000}, device_scale_factor=2, reduced_motion="reduce")
    page = ctx.new_page()
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(args.url, wait_until="domcontentloaded")
    page.get_by_role("button", name="Buka kasus", exact=True).wait_for(timeout=300000)

    def settle(title: str) -> None:
        page.get_by_role("heading", name=title, exact=True).wait_for(timeout=120000)
        page.wait_for_function("document.querySelectorAll('[data-testid=stSkeleton]').length === 0", timeout=120000)
        page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').wait_for(timeout=120000)
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(1500)
        assert page.locator('[data-testid="stException"]').count() == 0

    sidebar = page.locator('[data-testid="stSidebar"]')

    # Queue (S11)
    settle("Daftar prioritas")
    page.locator('[data-testid="stDataFrame"] canvas').first.wait_for()
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / "queue.png"))

    # Case PK00017 (S06): tall viewport so the reasons, Rp breakdown and payment chart are all visible.
    sidebar.get_by_text("Kasus perusahaan", exact=True).click()
    settle("Kasus perusahaan")
    box = page.locator('[data-testid="stSelectbox"]').filter(has_text="Pilih perusahaan")
    box.locator("input").click()
    page.keyboard.type("PT Baru Sejahtera")
    page.get_by_role("option", name=CASE, exact=True).click()
    settle("Kasus perusahaan")
    page.get_by_role("heading", name="PT Baru Sejahtera", exact=True).wait_for(timeout=60000)
    page.wait_for_function("document.querySelectorAll('[data-testid=stVegaLiteChart] canvas, [data-testid=stVegaLiteChart] svg').length >= 2", timeout=60000)
    page.set_viewport_size({"width": 1600, "height": 1360})
    page.wait_for_timeout(2500)
    page.locator('[data-testid="stMain"]').evaluate("e => e.scrollTop = 0")
    page.locator('[data-testid="stMain"]').screenshot(path=str(OUT / "case.png"))

    # Worker self-check (optional use)
    page.set_viewport_size({"width": 1600, "height": 1000})
    sidebar.get_by_text("Cek kepesertaan saya", exact=True).click()
    settle("Cek kepesertaan saya")
    page.locator('[data-testid="stMain"]').screenshot(path=str(OUT / "worker.png"))

    print("errors", errors)
    browser.close()
