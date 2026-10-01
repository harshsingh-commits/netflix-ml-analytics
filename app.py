"""Streamlit entry point for the Netflix catalog intelligence platform."""

from __future__ import annotations

import io
import logging
import pickle
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import CHART_COLORS, PROJECT_ROOT
from src.task3_rating_classification import ClassificationResult, train_rating_models
from src.task4_segmentation import (
    SegmentationResult,
    recommend_similar_titles,
    segment_content,
)
from src.task5_forecasting import ForecastResult, forecast_releases
from src.task6_analytics_engine import (
    AnalyticsBundle,
    analytics_csv_bundle,
    create_pdf_report,
    generate_analytics,
    save_plotly_png,
    shap_feature_importance,
)
from src.utils import (
    configure_logging,
    filter_titles,
    load_default_dataset,
    load_netflix_data,
    top_countries,
    top_genres,
)

configure_logging()
LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title="Netflix catalog intelligence", page_icon=":material/analytics:", layout="wide")

PAGES = ("Dashboard", "Rating classification", "Content segmentation", "Trend forecasting", "Analytics engine")


@st.cache_data(ttl="1h", max_entries=8, show_spinner=False)
def _load_uploaded_data(payload: bytes) -> pd.DataFrame:
    return load_netflix_data(payload)


def _clear_task_results_if_data_changed(frame: pd.DataFrame) -> None:
    signature_columns = ["title", "type", "release_year", "rating", "duration", "country", "listed_in", "date_added"]
    signature_frame = frame.reindex(columns=signature_columns).astype("string")
    signature = int(pd.util.hash_pandas_object(signature_frame, index=False).sum())
    if st.session_state.get("dataset_signature") != signature:
        for key in ("classification_result", "segmentation_result", "forecast_result", "shap_importance"):
            st.session_state.pop(key, None)
        st.session_state["dataset_signature"] = signature


def _apply_theme(mode: str) -> None:
    if mode == "Dark":
        background, panel, foreground = "#101820", "#1E293B", "#F8FAFC"
    else:
        background, panel, foreground = "#F5F8FC", "#FFFFFF", "#1E293B"
    st.html(
        f"""<style>
        .stApp {{ background: {background}; color: {foreground}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{ background: {panel}; }}
        [data-testid="stMetric"] {{ background: {panel}; border: 1px solid rgba(100,116,139,.18); padding: 14px; border-radius: 8px; }}
        [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 8px; }}
        .block-container {{ padding-top: 1.7rem; }}
        @media (max-width: 640px) {{ .block-container {{ padding-left: 1rem; padding-right: 1rem; }} }}
        </style>"""
    )


def _chart_layout(figure: go.Figure, mode: str) -> go.Figure:
    figure.update_layout(
        template="plotly_dark" if mode == "Dark" else "plotly_white",
        colorway=list(CHART_COLORS),
        margin={"l": 18, "r": 18, "t": 52, "b": 18},
        legend_title_text="",
        font={"family": "Aptos, Segoe UI, sans-serif", "size": 12},
    )
    return figure


def _load_data(uploaded: Any) -> tuple[pd.DataFrame | None, str]:
    if uploaded is not None:
        try:
            return _load_uploaded_data(uploaded.getvalue()), uploaded.name
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            st.sidebar.error(f"Could not read this CSV: {exc}")
            return None, uploaded.name
    try:
        local = load_default_dataset()
        if local is not None:
            return local, "data/netflix.csv"
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        LOGGER.exception("Local dataset could not be loaded")
        st.sidebar.error(f"Local dataset error: {exc}")
    return None, "No dataset loaded"


def _sidebar() -> tuple[str, str, str, pd.DataFrame | None]:
    with st.sidebar:
        st.markdown("## :material/movie: CATALOG / INTELLIGENCE")
        st.caption("Netflix content analytics")
        selected_page = st.selectbox("Select task", PAGES, key="selected_page")
        uploaded = st.file_uploader("Upload Netflix CSV", type=("csv",), help="Standard Netflix titles CSV or a compatible export.")
        theme = st.selectbox("Appearance", ("Light", "Dark"), index=0, key="appearance")
        data, source = _load_data(uploaded)
        if data is not None:
            st.caption(f"Source: {source} · {len(data):,} titles")
            genres = top_genres(data, limit=80).index.astype(str).tolist()
            ratings = sorted(data["rating"].dropna().astype(str).unique().tolist())
            countries = top_countries(data, limit=80).index.astype(str).tolist()
            types = sorted(data["type"].dropna().astype(str).unique().tolist())
            selected_genres = st.multiselect("Genre", genres, key="filter_genres")
            selected_ratings = st.multiselect("Rating", ratings, key="filter_ratings")
            selected_countries = st.multiselect("Country", countries, key="filter_countries")
            selected_types = st.multiselect("Content type", types, key="filter_types")
            years = pd.to_numeric(data["release_year"], errors="coerce").dropna()
            year_range: tuple[int, int] | None = None
            if not years.empty and int(years.min()) != int(years.max()):
                year_range = st.slider(
                    "Release year",
                    min_value=int(years.min()),
                    max_value=int(years.max()),
                    value=(int(years.min()), int(years.max())),
                    key="filter_years",
                )
            filtered = filter_titles(
                data,
                genres=selected_genres or None,
                ratings=selected_ratings or None,
                countries=selected_countries or None,
                years=year_range,
                content_types=selected_types or None,
            )
            st.caption(f"{len(filtered):,} titles match the current filters")
        else:
            filtered = None
            st.caption("Upload a CSV to unlock analysis.")
    _apply_theme(theme)
    return selected_page, source, theme, filtered


def _page_header(title: str, subtitle: str, icon: str) -> None:
    st.title(title, icon=icon)
    st.caption(subtitle)


def _render_empty_state() -> None:
    st.title("Netflix catalog intelligence", icon=":material/analytics:")
    st.markdown("### Bring your catalog into focus")
    st.write("Upload the Netflix titles CSV in the sidebar, or place `netflix.csv` in the project’s `data/` folder.")
    st.caption("Expected fields include title, type, release_year, rating, duration, country, listed_in, and date_added.")
    st.info("No example records or synthetic charts are shown. Results are generated from the dataset you provide.", icon=":material/database:")


def _render_dashboard(frame: pd.DataFrame, source: str, mode: str) -> None:
    _page_header("Catalog overview", f"Live view of {source} · filtered records only", ":material/dashboard:")
    unique_years = frame["release_year"].dropna()
    movie_share = frame["type"].eq("Movie").mean() * 100 if len(frame) else 0
    with st.container(horizontal=True):
        st.metric("Titles in view", f"{len(frame):,}", border=True)
        st.metric("Movies", f"{frame['type'].eq('Movie').sum():,}", border=True)
        st.metric("TV shows", f"{frame['type'].eq('Tv Show').sum():,}", border=True)
        st.metric("Movie share", f"{movie_share:.1f}%", border=True)
        st.metric("Release span", f"{int(unique_years.min())}–{int(unique_years.max())}" if not unique_years.empty else "Unavailable", border=True)

    left, right = st.columns(2)
    with left:
        genre_counts = top_genres(frame, limit=12).rename_axis("Genre").reset_index(name="Titles")
        figure = px.bar(genre_counts.sort_values("Titles"), x="Titles", y="Genre", orientation="h", title="Genre mix", text="Titles")
        st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="dashboard_genres")
    with right:
        type_counts = frame["type"].value_counts().rename_axis("Format").reset_index(name="Titles")
        figure = px.pie(type_counts, names="Format", values="Titles", hole=0.63, title="Format distribution")
        st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="dashboard_formats")
    release_counts = frame.dropna(subset=["release_year"]).groupby("release_year", as_index=False).size().rename(columns={"size": "Titles"})
    if not release_counts.empty:
        figure = px.area(release_counts, x="release_year", y="Titles", title="Titles by original release year")
        st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="dashboard_release_year")
    st.subheader("Dataset overview", icon=":material/table_chart:")
    st.dataframe(frame[["title", "type", "release_year", "rating", "duration", "country", "listed_in"]].head(100), width="stretch", hide_index=True)


def _render_classification(frame: pd.DataFrame, mode: str) -> None:
    _page_header("Rating classification", "Predict audience-rating categories from content metadata.", ":material/verified:")
    st.caption("Model evaluation uses a stratified 75/25 holdout. GridSearchCV tunes Random Forest on the training split.")
    if frame["rating"].dropna().nunique() < 2:
        st.warning("At least two non-empty rating categories are required.")
        return
    tune = st.checkbox("Run Random Forest GridSearchCV", value=True, help="Adds a small cross-validated parameter search.")
    if st.button("Train and compare models", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Training classification models and evaluating the holdout set…"):
            try:
                result = train_rating_models(frame, tune=tune)
                st.session_state["classification_result"] = result
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return
    result: ClassificationResult | None = st.session_state.get("classification_result")
    if result is None:
        st.info("Choose whether to tune, then train the models to see measured results.", icon=":material/model_training:")
        return
    st.success(f"Best holdout model: {result.best_model_name} · {result.train_rows:,} train / {result.test_rows:,} test rows")
    if result.note:
        st.caption(result.note)
    metrics = result.metrics.iloc[0]
    with st.container(horizontal=True):
        st.metric("Accuracy", f"{metrics['Accuracy']:.3f}", border=True)
        st.metric("Weighted precision", f"{metrics['Precision']:.3f}", border=True)
        st.metric("Weighted recall", f"{metrics['Recall']:.3f}", border=True)
        st.metric("Weighted F1", f"{metrics['F1 score']:.3f}", border=True)
    left, right = st.columns(2)
    with left:
        score_long = result.metrics.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 score"], var_name="Metric", value_name="Score")
        figure = px.bar(score_long, x="Model", y="Score", color="Metric", barmode="group", title="Holdout model comparison")
        st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="classification_comparison")
    with right:
        selected_model = st.selectbox("Confusion matrix", list(result.confusion_matrices), key="confusion_model")
        matrix = result.confusion_matrices[selected_model]
        figure = px.imshow(matrix, text_auto=True, aspect="auto", color_continuous_scale="Blues", labels={"x": "Predicted rating", "y": "Actual rating", "color": "Titles"}, title=f"{selected_model} · confusion matrix")
        st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="classification_confusion")
    if not result.feature_importance.empty:
        figure = px.bar(result.feature_importance.head(15).sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Feature importance")
        st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="classification_importance")
    if result.grid_search_best_params:
        st.caption(f"Best GridSearchCV parameters: {result.grid_search_best_params}")
    st.download_button("Download metrics CSV", result.metrics.to_csv(index=False), file_name="rating_model_metrics.csv", mime="text/csv", icon=":material/download:")
    model_buffer = io.BytesIO()
    pickle.dump(result.best_model, model_buffer)
    st.download_button("Download best model", model_buffer.getvalue(), file_name="netflix_rating_model.pkl", mime="application/octet-stream", icon=":material/download:")


def _render_segmentation(frame: pd.DataFrame, mode: str) -> None:
    _page_header("Content segmentation", "Discover groups from format, genre, country, release year, and duration.", ":material/hub:")
    st.caption("Clustering is capped at 2,500 sampled titles for responsive PCA/t-SNE and agglomerative runs.")
    if st.button("Run segmentation", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Encoding title features and comparing cluster counts…"):
            try:
                st.session_state["segmentation_result"] = segment_content(frame)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return
    result: SegmentationResult | None = st.session_state.get("segmentation_result")
    if result is None:
        st.info("Run segmentation to see the measured cluster structure.", icon=":material/scatter_plot:")
        return
    st.caption(f"Silhouette-selected K: {result.titles['cluster'].nunique()} · PCA variance explained: {result.pca_variance:.1%} · {len(result.titles):,} titles")
    left, right = st.columns(2)
    with left:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=result.selection["Clusters"], y=result.selection["Inertia"], mode="lines+markers", name="Inertia"))
        figure.update_layout(title="Elbow method", xaxis_title="Clusters (K)", yaxis_title="Inertia")
        st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="segmentation_elbow")
    with right:
        figure = px.line(result.selection, x="Clusters", y="Silhouette", markers=True, title="Silhouette score by K")
        st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="segmentation_silhouette")
    plot_method = st.segmented_control("Projection", ["PCA", "t-SNE"], default="PCA", key="segmentation_projection")
    x_column, y_column = ("pca_1", "pca_2") if plot_method != "t-SNE" else ("tsne_1", "tsne_2")
    figure = px.scatter(
        result.titles,
        x=x_column,
        y=y_column,
        color=result.titles["cluster"].astype(str),
        symbol="type",
        hover_name="title",
        hover_data=["type", "release_year", "rating", "listed_in"],
        title=f"{plot_method} projection of KMeans segments",
        labels={"color": "Cluster"},
    )
    st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="segmentation_scatter")
    cluster_counts = result.titles["cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Titles")
    figure = px.bar(cluster_counts, x="Cluster", y="Titles", title="Cluster distribution", text="Titles")
    st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="segmentation_distribution")
    st.subheader("Cluster profiles")
    st.dataframe(result.insights, width="stretch", hide_index=True)
    title_options = result.titles["title"].dropna().astype(str).tolist()
    chosen_title = st.selectbox("Find titles similar to", title_options, key="recommendation_title")
    if chosen_title:
        recommendations = recommend_similar_titles(result, chosen_title)
        st.markdown(f"**Similar to {chosen_title}**")
        st.dataframe(recommendations[["title", "type", "release_year", "rating", "listed_in"]], width="stretch", hide_index=True)


def _render_forecasting(frame: pd.DataFrame, mode: str) -> None:
    _page_header("Trend forecasting", "Forecast catalog additions from observed date_added history.", ":material/query_stats:")
    st.caption("Forecasts model titles added to the catalog, not original production output. Release-year counts are shown separately.")
    if frame["date_added_dt"].notna().sum() == 0:
        annual = frame.dropna(subset=["release_year"]).groupby("release_year", as_index=False).size().rename(columns={"size": "Titles"})
        if not annual.empty:
            st.plotly_chart(_chart_layout(px.line(annual, x="release_year", y="Titles", markers=True, title="Observed titles by original release year"), mode), width="stretch", key="annual_only_trend")
        st.warning("Monthly forecasting needs actual date_added values; no monthly history was synthesized.")
        return
    horizon = st.segmented_control("Forecast horizon", [12, 24], default=12, format_func=lambda value: f"{value} months", key="forecast_horizon")
    if st.button("Evaluate and forecast", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Backtesting candidate models and generating forecasts…"):
            try:
                st.session_state["forecast_result"] = forecast_releases(frame)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return
    result: ForecastResult | None = st.session_state.get("forecast_result")
    if result is None:
        st.info("Run the forecast to evaluate models and generate both requested horizons.", icon=":material/timeline:")
        return
    historical = result.monthly_history.rename("Observed").reset_index()
    historical.columns = ["Date", "Observed"]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=historical["Date"], y=historical["Observed"], mode="lines", name="Observed", line={"color": "#1E293B", "width": 2}))
    selected_forecast = result.forecasts[result.forecasts["Horizon"] == horizon]
    for model_name, values in selected_forecast.groupby("Model"):
        figure.add_trace(go.Scatter(x=values["Date"], y=values["Forecast"], mode="lines", name=model_name, line={"dash": "dash"}))
    figure.update_layout(title=f"Observed and forecast catalog additions · next {horizon} months", xaxis_title="Month", yaxis_title="Titles added")
    st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="forecast_trend")
    left, right = st.columns(2)
    with left:
        st.subheader("Model comparison")
        st.dataframe(result.metrics, width="stretch", hide_index=True)
    with right:
        annual = result.annual_releases.rename_axis("Release year").reset_index(name="Titles")
        figure = px.bar(annual, x="Release year", y="Titles", title="Observed content by original release year")
        st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="forecast_annual")
    if result.unavailable_models:
        st.caption("Some models could not run: " + "; ".join(result.unavailable_models))
    st.subheader("Business conclusions")
    for insight in result.insights:
        st.markdown(f"- {insight}")
    st.download_button("Download forecasts CSV", result.forecasts.to_csv(index=False), file_name="release_forecasts.csv", mime="text/csv", icon=":material/download:")


def _render_analytics(frame: pd.DataFrame, mode: str) -> None:
    _page_header("Analytics engine", "Automated EDA, growth signals, explainability, and exportable reports.", ":material/auto_awesome:")
    bundle: AnalyticsBundle = generate_analytics(frame)
    for insight in bundle.insights:
        st.markdown(f"- {insight}")
    st.subheader("Catalog summary")
    st.dataframe(bundle.summary, width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(_chart_layout(px.bar(bundle.genres.head(12).sort_values("Titles"), x="Titles", y="Genre", orientation="h", title="Top genres"), mode), width="stretch", key="analytics_genres")
        st.plotly_chart(_chart_layout(px.bar(bundle.countries.head(12).sort_values("Titles"), x="Titles", y="Country", orientation="h", title="Top credited countries"), mode), width="stretch", key="analytics_countries")
    with right:
        if not bundle.release_trend.empty:
            st.plotly_chart(_chart_layout(px.line(bundle.release_trend, x="added_year", y="Titles_added", markers=True, title="Catalog additions by year"), mode), width="stretch", key="analytics_release_trend")
        if not bundle.ratings.empty:
            st.plotly_chart(_chart_layout(px.bar(bundle.ratings.head(12), x="Rating", y="Titles", title="Audience rating distribution"), mode), width="stretch", key="analytics_ratings")
    if not bundle.fastest_growing_genres.empty:
        st.subheader("Fastest-growing genres")
        st.dataframe(bundle.fastest_growing_genres, width="stretch", hide_index=True)
    with st.expander("Missing values and numeric correlations"):
        st.dataframe(bundle.missing, width="stretch", hide_index=True)
        correlation = bundle.correlation
        if not correlation.empty:
            figure = px.imshow(correlation, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Feature correlation")
            st.plotly_chart(_chart_layout(figure, mode), width="stretch", key="analytics_correlation")
    classification: ClassificationResult | None = st.session_state.get("classification_result")
    if classification is not None:
        with st.expander("Model explainability (SHAP)"):
            if st.button("Calculate SHAP feature importance", key="run_shap"):
                try:
                    st.session_state["shap_importance"] = shap_feature_importance(classification.best_model, frame)
                except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                    st.warning(f"SHAP could not explain this fitted model: {exc}")
            shap_result = st.session_state.get("shap_importance")
            if shap_result is not None:
                st.dataframe(shap_result, width="stretch", hide_index=True)
    st.subheader("Export reports")
    left, right = st.columns(2)
    with left:
        st.download_button("Download PDF report", create_pdf_report(bundle), file_name="netflix_analytics_report.pdf", mime="application/pdf", icon=":material/picture_as_pdf:")
        st.download_button("Download EDA CSV bundle", analytics_csv_bundle(bundle), file_name="netflix_eda_results.zip", mime="application/zip", icon=":material/table_view:")
    with right:
        chart = px.bar(bundle.genres.head(12).sort_values("Titles"), x="Titles", y="Genre", orientation="h", title="Top genres")
        _chart_layout(chart, mode)
        if st.button("Save top-genres PNG", icon=":material/image:"):
            try:
                path = save_plotly_png(chart, "top_genres.png")
                st.success(f"Saved {path.relative_to(PROJECT_ROOT)}")
                st.download_button("Download top-genres PNG", path.read_bytes(), file_name="top_genres.png", mime="image/png", key="download_genre_png")
            except RuntimeError as exc:
                st.error(str(exc))


page, source, appearance, filtered_data = _sidebar()
if filtered_data is None:
    _render_empty_state()
elif filtered_data.empty:
    st.warning("No titles match the current sidebar filters. Broaden one or more filters to continue.")
else:
    _clear_task_results_if_data_changed(filtered_data)
    page_renderer = {
        "Dashboard": _render_dashboard,
        "Rating classification": _render_classification,
        "Content segmentation": _render_segmentation,
        "Trend forecasting": _render_forecasting,
        "Analytics engine": _render_analytics,
    }
    if page == "Dashboard":
        page_renderer[page](filtered_data, source, appearance)
    else:
        page_renderer[page](filtered_data, appearance)