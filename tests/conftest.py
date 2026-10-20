"""Tests never write the local case file: the app uses an in-memory case store per session."""
import os

os.environ.setdefault("TEPAT_CASES", "sandbox")
