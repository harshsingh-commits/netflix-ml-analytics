"""Shared project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"
MODEL_DIR = OUTPUT_DIR / "models"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"

for directory in (DATA_DIR, CHART_DIR, REPORT_DIR, MODEL_DIR, SCREENSHOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
RATING_LABELS = ("TV-Y", "TV-Y7", "TV-G", "TV-PG", "TV-14", "TV-MA", "R", "PG-13", "PG")
CHART_COLORS = ("#2563EB", "#10B981", "#F59E0B", "#EF4444", "#0F766E", "#64748B")