"""Data loading, cleaning, and shared feature engineering."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from src.config import DATA_DIR, PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
CANONICAL_COLUMNS = (
    "show_id", "type", "title", "director", "cast", "country", "date_added",
    "release_year", "rating", "duration", "listed_in", "description",
)
COLUMN_ALIASES = {
    "id": "show_id", "content_type": "type", "category": "type", "name": "title",
    "genres": "listed_in", "genre": "listed_in", "year": "release_year",
    "added_date": "date_added", "audience_rating": "rating",
}


def configure_logging() -> None:
    """Configure concise application logging once per process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [re.sub(r"\s+", "_", str(column).strip().lower()) for column in result.columns]
    result = result.rename(columns={key: value for key, value in COLUMN_ALIASES.items() if key in result.columns})
    return result


def load_netflix_data(source: str | Path | BinaryIO | bytes) -> pd.DataFrame:
    """Read and clean a Netflix titles CSV from a path, uploaded file, or bytes."""
    try:
        if isinstance(source, bytes):
            from io import BytesIO

            frame = pd.read_csv(BytesIO(source))
        else:
            frame = pd.read_csv(source)
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        LOGGER.warning("Default CSV parsing failed; retrying with a permissive encoding: %s", exc)
        if isinstance(source, bytes):
            from io import BytesIO

            frame = pd.read_csv(BytesIO(source), encoding="latin-1", on_bad_lines="skip")
        else:
            frame = pd.read_csv(source, encoding="latin-1", on_bad_lines="skip")
    return clean_netflix_data(frame)


def load_default_dataset() -> pd.DataFrame | None:
    """Load data/netflix.csv when present; return None when it is not supplied."""
    candidates = (DATA_DIR / "netflix.csv", DATA_DIR / "netflix_titles.csv", PROJECT_ROOT / "netflix.csv", PROJECT_ROOT / "netflix_titles.csv")
    for path in candidates:
        if path.is_file():
            LOGGER.info("Loading local dataset: %s", path)
            return load_netflix_data(path)
    root_csvs = list(PROJECT_ROOT.glob("*.csv"))
    netflix_csvs = [path for path in root_csvs if "netflix" in path.stem.casefold()]
    if len(netflix_csvs) == 1:
        return load_netflix_data(netflix_csvs[0])
    if len(root_csvs) == 1:
        return load_netflix_data(root_csvs[0])
    return None


def clean_netflix_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the common Netflix schema and create analysis-ready features."""
    if frame.empty:
        raise ValueError("The uploaded CSV has no data rows.")
    cleaned = _canonicalize_columns(frame)
    if not len(cleaned.columns):
        raise ValueError("The uploaded CSV has no recognizable columns.")
    for column in CANONICAL_COLUMNS:
        if column not in cleaned:
            cleaned[column] = pd.NA

    duplicate_keys = [column for column in ("title", "type", "release_year") if column in cleaned]
    duplicates_removed = int(cleaned.duplicated(subset=duplicate_keys).sum()) if duplicate_keys else 0
    cleaned = cleaned.drop_duplicates(subset=duplicate_keys)
    cleaned["type"] = cleaned["type"].astype("string").str.strip().str.title().fillna("Unknown")
    for column in ("title", "director", "cast", "country", "rating", "duration", "listed_in", "description"):
        cleaned[column] = cleaned[column].astype("string").str.strip()
        cleaned[column] = cleaned[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    cleaned["release_year"] = pd.to_numeric(cleaned["release_year"], errors="coerce")
    cleaned["date_added_dt"] = pd.to_datetime(cleaned["date_added"], errors="coerce", format="mixed")
    cleaned["date_added"] = cleaned["date_added_dt"].dt.strftime("%B %-d, %Y")
    cleaned["date_added"] = cleaned["date_added"].fillna("Unknown")
    cleaned["director_count"] = cleaned["director"].map(_list_count)
    cleaned["cast_count"] = cleaned["cast"].map(_list_count)
    cleaned["genre_list"] = cleaned["listed_in"].map(_split_list)
    cleaned["genre_count"] = cleaned["genre_list"].map(len)
    cleaned["country_list"] = cleaned["country"].map(_split_list)
    cleaned["country_primary"] = cleaned["country_list"].map(lambda values: values[0] if values else "Unknown")
    duration_parts = cleaned["duration"].map(parse_duration)
    cleaned["duration_value"] = duration_parts.map(lambda value: value[0])
    cleaned["duration_unit"] = duration_parts.map(lambda value: value[1])
    cleaned["release_decade"] = (cleaned["release_year"] // 10 * 10).astype("Int64")
    cleaned["added_year"] = cleaned["date_added_dt"].dt.year.astype("Int64")
    cleaned["added_month"] = cleaned["date_added_dt"].dt.month.astype("Int64")
    cleaned = cleaned.reset_index(drop=True)
    cleaned.attrs["duplicates_removed"] = duplicates_removed
    return cleaned


def _split_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _list_count(value: object) -> int:
    return len(_split_list(value))


def parse_duration(value: object) -> tuple[float, str]:
    """Extract the first numeric duration and its unit from Netflix duration text."""
    if pd.isna(value):
        return (np.nan, "Unknown")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(min|season|seasons)", str(value), re.IGNORECASE)
    if not match:
        return (np.nan, "Unknown")
    unit = "min" if match.group(2).lower() == "min" else "seasons"
    return (float(match.group(1)), unit)


def top_genres(frame: pd.DataFrame, limit: int = 10) -> pd.Series:
    """Count genre tags across titles, including multi-genre rows."""
    return frame["genre_list"].explode().dropna().value_counts().head(limit)


def top_countries(frame: pd.DataFrame, limit: int = 10) -> pd.Series:
    """Count country tags across titles, including co-productions."""
    return frame["country_list"].explode().dropna().value_counts().head(limit)


def filter_titles(
    frame: pd.DataFrame,
    genres: list[str] | None = None,
    ratings: list[str] | None = None,
    countries: list[str] | None = None,
    years: tuple[int, int] | None = None,
    content_types: list[str] | None = None,
) -> pd.DataFrame:
    """Apply optional dashboard filters without mutating the source frame."""
    result = frame
    if genres:
        result = result[result["genre_list"].map(lambda values: any(item in values for item in genres))]
    if ratings:
        result = result[result["rating"].isin(ratings)]
    if countries:
        result = result[result["country_list"].map(lambda values: any(item in values for item in countries))]
    if years:
        result = result[result["release_year"].between(years[0], years[1])]
    if content_types:
        result = result[result["type"].isin(content_types)]
    return result