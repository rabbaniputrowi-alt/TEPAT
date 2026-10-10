"""Crop the case capture to the evidence column (reasons + documents) for slide S06.
The payment chart on the right is flat for this employer (no late payments) and reads as broken
when shrunk, so the slide shows the evidence column only.
Run: uv run --with pillow python proposal/crop.py
"""
from pathlib import Path

from PIL import Image

A = Path(__file__).resolve().parent / "assets"
im = Image.open(A / "case.png")
print("case.png", im.size)
# 2x capture: the left evidence column spans x 84..1520 and the reasons box starts near y 970.
im.crop((70, 958, 1536, 2240)).save(A / "case-reasons.png", optimize=True)
print("case-reasons.png", Image.open(A / "case-reasons.png").size)
