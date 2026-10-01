"""Automated EDA, business insights, explainability, and report exports."""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import CHART_DIR, REPORT_DIR
from src.task3_rating_classification import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.utils import top_countries, top_genres

LOGGER = logging.getLogger(__name__)


@dataclass
class AnalyticsBundle:
    """Data tables and narrative insights generated from one cleaned dataset."""

    summary: pd.DataFrame
    missing: pd.DataFrame
    correlation: pd.DataFrame
    genres: pd.DataFrame
    countries: pd.DataFrame
    release_trend: pd.DataFrame
    ratings: pd.DataFrame
    rating_trend: pd.DataFrame
    fastest_growing_genres: pd.DataFrame
    insights: list[str]


def generate_analytics(frame: pd.DataFrame) -> AnalyticsBundle:
    """Create descriptive EDA tables and plain-language business insights."""
    raw_columns = [column for column in frame.columns if column not in {"genre_list", "country_list", "date_added_dt"}]
    missing = pd.DataFrame(
        {
            "Column": raw_columns,
            "Missing values": [
                int(frame["date_added_dt"].isna().sum()) if column == "date_added" else int(frame[column].isna().sum())
                for column in raw_columns
            ],
            "Missing %": [
                round(float(frame["date_added_dt"].isna().mean() * 100), 2)
                if column == "date_added"
                else round(float(frame[column].isna().mean() * 100), 2)
                for column in raw_columns
            ],
        }
    ).sort_values("Missing values", ascending=False)
    summary = pd.DataFrame(
        [
            ("Titles", len(frame)),
            ("Unique titles", int(frame["title"].nunique(dropna=True))),
            ("Movie titles", int(frame["type"].eq("Movie").sum())),
            ("TV shows", int(frame["type"].eq("Tv Show").sum())),
            ("Unique ratings", int(frame["rating"].nunique(dropna=True))),
            ("First release year", _safe_min(frame["release_year"])),
            ("Latest release year", _safe_max(frame["release_year"])),
            ("Rows with catalog-add dates", int(frame["date_added_dt"].notna().sum())),
            ("Duplicate records removed", int(frame.attrs.get("duplicates_removed", 0))),
        ],
        columns=["Metric", "Value"],
    )
    numeric_columns = [column for column in (*NUMERIC_FEATURES, "added_year", "added_month") if column in frame]
    correlation = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").corr().fillna(0)
    genres = top_genres(frame, 20).rename_axis("Genre").reset_index(name="Titles")
    countries = top_countries(frame, 20).rename_axis("Country").reset_index(name="Titles")
    ratings = frame["rating"].fillna("Unknown").value_counts().rename_axis("Rating").reset_index(name="Titles")
    release_trend = (
        frame.dropna(subset=["added_year"])
        .groupby("added_year", as_index=False)
        .agg(Titles_added=("title", "size"))
        .sort_values("added_year")
    )
    rating_trend = (
        frame.dropna(subset=["added_year"])
        .groupby(["added_year", "rating"], dropna=False)
        .size()
        .rename("Titles")
        .reset_index()
        .sort_values("added_year")
    )
    growth = _genre_growth(frame)
    insights = _build_insights(frame, genres, countries, ratings, release_trend, growth)
    return AnalyticsBundle(
        summary=summary,
        missing=missing,
        correlation=correlation,
        genres=genres,
        countries=countries,
        release_trend=release_trend,
        ratings=ratings,
        rating_trend=rating_trend,
        fastest_growing_genres=growth,
        insights=insights,
    )


def _safe_min(values: pd.Series) -> int | str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return int(values.min()) if not values.empty else "Unavailable"


def _safe_max(values: pd.Series) -> int | str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return int(values.max()) if not values.empty else "Unavailable"


def _genre_growth(frame: pd.DataFrame) -> pd.DataFrame:
    dated = frame.dropna(subset=["added_year"])
    years = sorted(dated["added_year"].astype(int).unique())
    if len(years) < 2:
        return pd.DataFrame(columns=["Genre", "Earlier titles", "Recent titles", "Growth %"])
    split_year = years[max(0, len(years) // 2 - 1)]
    first = dated[dated["added_year"] <= split_year][["added_year", "genre_list"]].explode("genre_list")
    recent = dated[dated["added_year"] > split_year][["added_year", "genre_list"]].explode("genre_list")
    earlier_counts = first["genre_list"].value_counts()
    recent_counts = recent["genre_list"].value_counts()
    table = pd.concat([earlier_counts.rename("Earlier titles"), recent_counts.rename("Recent titles")], axis=1).fillna(0)
    table = table[(table["Earlier titles"] >= 3) & (table["Recent titles"] > 0)]
    if table.empty:
        return pd.DataFrame(columns=["Genre", "Earlier titles", "Recent titles", "Growth %"])
    table["Growth %"] = (table["Recent titles"] / table["Earlier titles"] - 1) * 100
    return table.sort_values("Growth %", ascending=False).head(10).rename_axis("Genre").reset_index().round(1)


def _build_insights(
    frame: pd.DataFrame,
    genres: pd.DataFrame,
    countries: pd.DataFrame,
    ratings: pd.DataFrame,
    release_trend: pd.DataFrame,
    growth: pd.DataFrame,
) -> list[str]:
    insights: list[str] = []
    if not genres.empty:
        insights.append(f"{genres.iloc[0]['Genre']} is the largest genre group with {genres.iloc[0]['Titles']:,} tagged titles.")
    if not frame["type"].value_counts().empty:
        type_name = frame["type"].value_counts().index[0]
        type_share = frame["type"].value_counts(normalize=True).iloc[0] * 100
        insights.append(f"{type_name} is the most common format ({type_share:.1f}% of the catalog).")
    if not ratings.empty:
        insights.append(f"{ratings.iloc[0]['Rating']} is the most frequent audience rating ({ratings.iloc[0]['Titles']:,} titles).")
    if not countries.empty:
        insights.append(f"{countries.iloc[0]['Country']} appears most often in country credits ({countries.iloc[0]['Titles']:,} titles).")
    if not growth.empty:
        item = growth.iloc[0]
        insights.append(f"{item['Genre']} grew fastest across the observed catalog-add periods ({item['Growth %']:+.1f}%).")
    if len(release_trend) >= 2:
        latest = release_trend.iloc[-1]
        previous = release_trend.iloc[-2]
        insights.append(
            f"Catalog additions changed {((latest['Titles_added'] / previous['Titles_added']) - 1) * 100 if previous['Titles_added'] else 0:+.1f}% "
            f"from {int(previous['added_year'])} to {int(latest['added_year'])}."
        )
    if not frame["release_year"].dropna().empty:
        median_year = int(frame["release_year"].median())
        insights.append(f"The median content release year is {median_year}; this is distinct from when titles were added to Netflix.")
    return insights


def shap_feature_importance(model: Pipeline, frame: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
    """Compute a bounded mean absolute SHAP ranking for a fitted tree pipeline."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Install dependencies from requirements.txt to enable explainability.") from exc
    feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    raw = frame.reindex(columns=feature_columns).sample(min(sample_size, len(frame)), random_state=42)
    transformed = model.named_steps["preprocessor"].transform(raw)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = model.named_steps["preprocessor"].get_feature_names_out()
    estimator = model.named_steps["classifier"]
    background = transformed[: min(50, len(transformed))]
    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
        values = np.asarray(explainer.shap_values(transformed, check_additivity=False))
    else:
        explainer = shap.Explainer(estimator, background, feature_names=names)
        values = np.asarray(explainer(transformed).values)
    if values.ndim == 3:
        if values.shape[1] == len(names):
            importance = np.abs(values).mean(axis=(0, 2))
        else:
            importance = np.abs(values).mean(axis=(1, 2))
    elif values.ndim == 2:
        importance = np.abs(values).mean(axis=0)
    else:
        raise ValueError("SHAP returned an unsupported importance shape.")
    return pd.DataFrame({"Feature": names, "Mean |SHAP value|": importance}).sort_values(
        "Mean |SHAP value|", ascending=False
    ).head(30).reset_index(drop=True)


def analytics_csv_bundle(bundle: AnalyticsBundle) -> bytes:
    """Package the EDA tables as a downloadable ZIP of CSV result files."""
    tables = {
        "summary.csv": bundle.summary,
        "missing_values.csv": bundle.missing,
        "correlations.csv": bundle.correlation.reset_index(names="Feature"),
        "genres.csv": bundle.genres,
        "countries.csv": bundle.countries,
        "release_trend.csv": bundle.release_trend,
        "ratings.csv": bundle.ratings,
        "rating_trend.csv": bundle.rating_trend,
        "fastest_growing_genres.csv": bundle.fastest_growing_genres,
        "insights.csv": pd.DataFrame({"Insight": bundle.insights}),
    }
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, table in tables.items():
            archive.writestr(filename, table.to_csv(index=False))
    return payload.getvalue()


def create_pdf_report(bundle: AnalyticsBundle, title: str = "Netflix catalog analytics report") -> bytes:
    """Create a concise PDF report from computed analytics tables and insights."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("ReportLab is not installed. Install dependencies from requirements.txt to export PDF reports.") from exc
    payload = io.BytesIO()
    document = SimpleDocTemplate(payload, pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch)
    styles = getSampleStyleSheet()
    story: list[Any] = [Paragraph(title, styles["Title"]), Spacer(1, 14)]
    story.append(Paragraph("Executive insights", styles["Heading2"]))
    for insight in bundle.insights:
        story.append(Paragraph(f"&#8226; {insight}", styles["BodyText"]))
        story.append(Spacer(1, 5))
    story.extend([Spacer(1, 10), Paragraph("Dataset summary", styles["Heading2"])])
    summary_rows = [bundle.summary.columns.tolist(), *bundle.summary.astype(str).values.tolist()]
    summary_table = Table(summary_rows, colWidths=[2.4 * inch, 3.5 * inch], repeatRows=1)
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 12), Paragraph("Leading genres", styles["Heading2"])])
    if not bundle.genres.empty:
        genre_rows = [bundle.genres.columns.tolist(), *bundle.genres.head(10).astype(str).values.tolist()]
        genre_table = Table(genre_rows, colWidths=[4.3 * inch, 1.6 * inch], repeatRows=1)
        genre_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(genre_table)
    document.build(story)
    return payload.getvalue()


def save_plotly_png(figure: Any, filename: str) -> Path:
    """Save one Plotly figure as PNG using the Kaleido image exporter."""
    path = CHART_DIR / filename
    try:
        figure.write_image(path, width=1400, height=850, scale=2)
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError("PNG export requires a working Plotly/Kaleido installation.") from exc
    return path


def save_pdf_report(bundle: AnalyticsBundle, filename: str = "netflix_analytics_report.pdf") -> Path:
    """Persist a generated PDF report under outputs/reports."""
    path = REPORT_DIR / filename
    path.write_bytes(create_pdf_report(bundle))
    return path