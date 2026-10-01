"""Capture real application screenshots for the project evaluation checklist."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCREENSHOT_DIR = PROJECT_ROOT / "outputs" / "screenshots"


def select_task(page: Page, task_name: str) -> None:
    page.get_by_role("combobox").first.click()
    page.get_by_role("option", name=task_name, exact=True).click()
    page.locator("[data-testid='stAppViewContainer']").wait_for(state="visible")


def capture(page: Page, name: str, full_page: bool = True) -> None:
    page.wait_for_function(
        """() => {
            const charts = Array.from(document.querySelectorAll('[data-testid="stPlotlyChart"]'));
            return charts.length > 0 && charts.every(chart => chart.querySelector('svg'));
        }""",
        timeout=60_000,
    )
    page.screenshot(path=SCREENSHOT_DIR / f"{name}.png", full_page=full_page, animations="disabled")
    print(f"Saved outputs/screenshots/{name}.png")


def run_action(page: Page, label: str, completion_text: str | None = None, timeout_ms: int = 240_000) -> None:
    page.get_by_role("button", name=label).click(timeout=15_000)
    if completion_text:
        page.get_by_text(completion_text, exact=False).wait_for(timeout=timeout_ms)
    else:
        page.locator("[data-testid='stAppViewContainer']").wait_for(state="visible", timeout=timeout_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Real Netflix titles CSV to upload into the running app.")
    parser.add_argument("--url", default="http://localhost:8501", help="URL of the running Streamlit app.")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_file():
        parser.error(f"Dataset does not exist: {dataset}")
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(args.url, wait_until="domcontentloaded", timeout=60_000)
        page.locator("input[type='file']").set_input_files(dataset)
        page.get_by_text("Source:", exact=False).wait_for(timeout=60_000)

        select_task(page, "Dashboard")
        capture(page, "dashboard_home", full_page=False)
        capture(page, "dataset_analysis")

        select_task(page, "Rating classification")
        run_action(page, "Train and compare models", "Best holdout model")
        capture(page, "rating_prediction")

        select_task(page, "Content segmentation")
        run_action(page, "Run segmentation", "Silhouette-selected K", timeout_ms=240_000)
        capture(page, "clustering_results")

        select_task(page, "Trend forecasting")
        try:
            run_action(page, "Evaluate and forecast", "Observed and forecast catalog additions", timeout_ms=240_000)
            capture(page, "forecasting_results")
        except PlaywrightTimeoutError:
            if page.get_by_text("Monthly forecasting needs actual date_added values", exact=False).count():
                page.screenshot(path=SCREENSHOT_DIR / "forecasting_results.png", full_page=True, animations="disabled")
            else:
                raise

        select_task(page, "Analytics engine")
        page.get_by_text("Catalog summary", exact=True).wait_for(timeout=60_000)
        capture(page, "analytics_engine")
        capture(page, "final_insights")
        browser.close()


if __name__ == "__main__":
    main()