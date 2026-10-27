"""Capture the public app's Daftar prioritas for the Project Final deck (slide 5, Demo).

Run: uv run --with playwright python proposal/project_final/capture_live.py
Anonymous visitor: fresh headless Chrome profile against https://tepat-jkp.streamlit.app.
Wakes the app if Community Cloud put it to sleep.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "https://tepat-jkp.streamlit.app/"
OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="C:/Program Files/Google/Chrome/Application/chrome.exe", headless=True)
    ctx = browser.new_context(viewport={"width": 1600, "height": 1060}, device_scale_factor=2, reduced_motion="reduce")
    page = ctx.new_page()
    page.goto(URL, wait_until="domcontentloaded", timeout=120000)
    page.wait_for_timeout(8000)
    wake = page.get_by_role("button", name="Yes, get this app back up!")
    if wake.count():
        print("app was asleep, waking")
        wake.click()
        page.wait_for_timeout(5000)
    # Community Cloud serves the app inside an iframe; work on the frame that holds stApp.
    frame = None
    for _ in range(240):
        for f in page.frames:
            try:
                if f.locator('[data-testid="stApp"]').count():
                    frame = f
                    break
            except Exception:
                pass
        if frame:
            break
        page.wait_for_timeout(1000)
    if frame is None:
        page.screenshot(path=str(OUT / "live_debug.png"))
        sys.exit("stApp not found")
    frame.get_by_text("Daftar prioritas").first.wait_for(timeout=300000)
    frame.wait_for_function("document.querySelectorAll('[data-testid=stSkeleton]').length === 0", timeout=300000)
    frame.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').wait_for(timeout=300000)
    frame.evaluate("document.fonts.ready")
    page.wait_for_timeout(4000)
    exc = frame.locator('[data-testid="stException"]').count()
    print("exceptions:", exc)
    # Screenshot just the app frame area (no Community Cloud chrome).
    el = page.locator("iframe").first if page.locator("iframe").count() else None
    if el is not None:
        el.screenshot(path=str(OUT / "live_queue.png"))
    else:
        page.screenshot(path=str(OUT / "live_queue.png"))
    print("saved", OUT / "live_queue.png")
    browser.close()
