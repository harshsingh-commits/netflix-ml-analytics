"""Netflix content segmentation, diagnostics, and similarity recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE

LOGGER = logging.getLogger(__name__)
NUMERIC_FEATURES = ("release_year", "director_count", "cast_count", "genre_count", "duration_value")


@dataclass
class SegmentationResult:
    """Cluster assignments, selection diagnostics, and dimensionality reductions."""

    titles: pd.DataFrame
    selection: pd.DataFrame
    insights: pd.DataFrame
    pca_variance: float
    feature_matrix: np.ndarray
    sampled_indices: np.ndarray


def _cluster_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric = frame.reindex(columns=NUMERIC_FEATURES).apply(pd.to_numeric, errors="coerce")
    numeric = SimpleImputer(strategy="median", add_indicator=True).fit_transform(numeric)
    numeric = StandardScaler().fit_transform(numeric)

    genre_counts = frame["genre_list"].explode().value_counts()
    country_counts = frame["country_primary"].value_counts()
    common_genres = genre_counts.head(18).index.tolist()
    common_countries = country_counts.head(18).index.tolist()
    reduced = pd.DataFrame(
        {
            "type": frame["type"].fillna("Unknown").astype(str),
            "duration_unit": frame["duration_unit"].fillna("Unknown").astype(str),
            "country_group": frame["country_primary"].map(lambda value: value if value in common_countries else "Other"),
        },
        index=frame.index,
    )
    one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    categorical = one_hot.fit_transform(reduced)
    genre_features = np.column_stack(
        [frame["genre_list"].map(lambda values, genre=genre: float(genre in values)).to_numpy() for genre in common_genres]
    ) if common_genres else np.empty((len(frame), 0))
    names = [f"numeric_{index}" for index in range(numeric.shape[1])]
    names.extend(one_hot.get_feature_names_out(reduced.columns).tolist())
    names.extend(f"genre_{genre}" for genre in common_genres)
    return np.hstack([numeric, categorical, genre_features]).astype(np.float32), names


def segment_content(frame: pd.DataFrame, max_rows: int = 2500) -> SegmentationResult:
    """Fit KMeans and agglomerative models and select K using silhouette score."""
    if len(frame) < 4:
        raise ValueError("At least four titles are required for meaningful clustering.")
    if len(frame) > max_rows:
        sample_indices = frame.sample(max_rows, random_state=RANDOM_STATE).index.to_numpy()
        sampled = frame.loc[sample_indices].copy()
    else:
        sampled = frame.copy()
        sample_indices = sampled.index.to_numpy()
    matrix, _ = _cluster_features(sampled)
    max_k = min(8, len(sampled) - 1)
    diagnostics: list[dict[str, float | int]] = []
    for count in range(2, max_k + 1):
        candidate = KMeans(n_clusters=count, n_init=10, random_state=RANDOM_STATE)
        labels = candidate.fit_predict(matrix)
        diagnostics.append(
            {
                "Clusters": count,
                "Inertia": float(candidate.inertia_),
                "Silhouette": float(silhouette_score(matrix, labels, sample_size=min(1500, len(sampled)), random_state=RANDOM_STATE)),
            }
        )
    selection = pd.DataFrame(diagnostics)
    chosen_k = int(selection.sort_values("Silhouette", ascending=False).iloc[0]["Clusters"])
    kmeans_labels = KMeans(n_clusters=chosen_k, n_init=15, random_state=RANDOM_STATE).fit_predict(matrix)
    agglomerative_labels = AgglomerativeClustering(n_clusters=chosen_k, linkage="ward").fit_predict(matrix)
    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(matrix)
    pca_model = PCA(n_components=2, random_state=RANDOM_STATE).fit(matrix)
    titles = sampled.copy()
    titles["cluster"] = kmeans_labels
    titles["agglomerative_cluster"] = agglomerative_labels
    titles["pca_1"] = coordinates[:, 0]
    titles["pca_2"] = coordinates[:, 1]
    titles["tsne_1"] = np.nan
    titles["tsne_2"] = np.nan
    if len(sampled) >= 5:
        perplexity = min(30.0, max(2.0, (len(sampled) - 1) / 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=RANDOM_STATE)
        tsne_coordinates = tsne.fit_transform(matrix)
        titles["tsne_1"] = tsne_coordinates[:, 0]
        titles["tsne_2"] = tsne_coordinates[:, 1]
    insights = _cluster_insights(titles)
    return SegmentationResult(
        titles=titles.reset_index(drop=True),
        selection=selection,
        insights=insights,
        pca_variance=float(pca_model.explained_variance_ratio_.sum()),
        feature_matrix=matrix,
        sampled_indices=sample_indices,
    )


def _cluster_insights(titles: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster, group in titles.groupby("cluster", sort=True):
        genre_counts = group["genre_list"].explode().value_counts()
        content_counts = group["type"].value_counts()
        rows.append(
            {
                "Cluster": int(cluster),
                "Titles": len(group),
                "Dominant format": content_counts.index[0] if not content_counts.empty else "Unknown",
                "Format mix": ", ".join(f"{name} ({count})" for name, count in content_counts.head(3).items()),
                "Top genres": ", ".join(genre_counts.head(3).index.astype(str)),
                "Average release year": round(float(group["release_year"].mean()), 1),
            }
        )
    return pd.DataFrame(rows)


def recommend_similar_titles(
    result: SegmentationResult,
    title: str,
    count: int = 5,
) -> pd.DataFrame:
    """Find nearest titles in the selected title's assigned KMeans segment."""
    matches = result.titles.index[result.titles["title"].astype(str).str.casefold() == title.casefold()].tolist()
    if not matches:
        raise ValueError(f"'{title}' is not present in the clustered sample.")
    query_index = matches[0]
    query_cluster = result.titles.loc[query_index, "cluster"]
    cluster_indices = np.flatnonzero(result.titles["cluster"].to_numpy() == query_cluster)
    candidate_positions = cluster_indices[cluster_indices != query_index]
    if not len(candidate_positions):
        return result.titles.iloc[0:0]
    neighbors = NearestNeighbors(n_neighbors=min(count, len(candidate_positions)), metric="euclidean")
    neighbors.fit(result.feature_matrix[candidate_positions])
    _, indices = neighbors.kneighbors(result.feature_matrix[[query_index]])
    return result.titles.iloc[candidate_positions[indices[0]]].reset_index(drop=True)