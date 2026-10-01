"""Audience-rating classification pipelines and evaluation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE

LOGGER = logging.getLogger(__name__)
NUMERIC_FEATURES = ("release_year", "director_count", "cast_count", "genre_count", "duration_value")
CATEGORICAL_FEATURES = ("type", "country_primary", "duration_unit", "listed_in")


@dataclass
class ClassificationResult:
    """Serializable model outputs used by the dashboard and report engine."""

    metrics: pd.DataFrame
    confusion_matrices: dict[str, pd.DataFrame]
    best_model_name: str
    best_model: Pipeline
    feature_importance: pd.DataFrame
    labels: list[str]
    train_rows: int
    test_rows: int
    grid_search_best_params: dict[str, Any]
    note: str = ""


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
        ]
    )
    return ColumnTransformer(
        transformers=[("numeric", numeric, list(NUMERIC_FEATURES)), ("categorical", categorical, list(CATEGORICAL_FEATURES))],
        remainder="drop",
    )


def _pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("preprocessor", _preprocessor()), ("classifier", estimator)])


def _model_candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Logistic regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=RANDOM_STATE),
        "Random forest": RandomForestClassifier(
            n_estimators=180, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Decision tree": DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=2, random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except ImportError:
        LOGGER.info("XGBoost is not installed; the three sklearn classifiers remain available.")
    return candidates


def train_rating_models(frame: pd.DataFrame, tune: bool = True) -> ClassificationResult:
    """Train requested classifiers and evaluate them on a stratified holdout."""
    if "rating" not in frame:
        raise ValueError("Rating classification requires a rating column.")
    eligible = frame.dropna(subset=["rating"]).copy()
    eligible["rating"] = eligible["rating"].astype(str).str.strip()
    counts = eligible["rating"].value_counts()
    excluded_ratings = counts[counts < 2].index.tolist()
    eligible = eligible[eligible["rating"].isin(counts[counts >= 2].index)]
    if eligible["rating"].nunique() < 2 or len(eligible) < 8:
        raise ValueError("At least two rating categories with two or more examples each are needed.")

    features = eligible.reindex(columns=[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES])
    encoder = LabelEncoder()
    target = encoder.fit_transform(eligible["rating"])
    class_counts = pd.Series(target).value_counts()
    test_count = max(int(np.ceil(len(target) * 0.25)), len(encoder.classes_))
    if test_count >= len(target) or class_counts.min() < 2:
        raise ValueError("There are not enough rows per rating category for a stratified train/test split.")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    candidates = _model_candidates()
    fitted: dict[str, Pipeline] = {}
    rows: list[dict[str, float | str]] = []
    matrices: dict[str, pd.DataFrame] = {}
    for name, estimator in candidates.items():
        model = _pipeline(estimator)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        elapsed = time.perf_counter() - started
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision,
                "Recall": recall,
                "F1 score": f1,
                "Fit seconds": elapsed,
            }
        )
        matrix = confusion_matrix(y_test, predictions, labels=np.arange(len(encoder.classes_)))
        matrices[name] = pd.DataFrame(matrix, index=encoder.classes_, columns=encoder.classes_)
        fitted[name] = model

    scores = pd.DataFrame(rows).sort_values("F1 score", ascending=False).reset_index(drop=True)
    params: dict[str, Any] = {}
    if tune:
        min_class_size = int(pd.Series(y_train).value_counts().min())
        folds = min(3, min_class_size)
        if folds >= 2:
            search = GridSearchCV(
                _pipeline(RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE)),
                {
                    "classifier__n_estimators": [120, 220],
                    "classifier__max_depth": [None, 16],
                    "classifier__min_samples_leaf": [1, 3],
                },
                scoring="f1_weighted",
                cv=folds,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(x_train, y_train)
            tuned_predictions = search.best_estimator_.predict(x_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, tuned_predictions, average="weighted", zero_division=0
            )
            scores = pd.concat(
                [
                    scores,
                    pd.DataFrame(
                        [{
                            "Model": "Random forest (tuned)",
                            "Accuracy": accuracy_score(y_test, tuned_predictions),
                            "Precision": precision,
                            "Recall": recall,
                            "F1 score": f1,
                            "Fit seconds": np.nan,
                        }]
                    ),
                ],
                ignore_index=True,
            ).sort_values("F1 score", ascending=False).reset_index(drop=True)
            matrices["Random forest (tuned)"] = pd.DataFrame(
                confusion_matrix(y_test, tuned_predictions, labels=np.arange(len(encoder.classes_))),
                index=encoder.classes_,
                columns=encoder.classes_,
            )
            fitted["Random forest (tuned)"] = search.best_estimator_
            params = search.best_params_

    best_name = str(scores.iloc[0]["Model"])
    best_model = fitted[best_name]
    importance = get_feature_importance(best_model)
    note = "Ratings with fewer than two rows were excluded from training." if excluded_ratings else ""
    return ClassificationResult(
        metrics=scores,
        confusion_matrices=matrices,
        best_model_name=best_name,
        best_model=best_model,
        feature_importance=importance,
        labels=encoder.classes_.tolist(),
        train_rows=len(x_train),
        test_rows=len(x_test),
        grid_search_best_params=params,
        note=note,
    )


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Return tree importance or coefficient magnitude using encoded names."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["classifier"]
    try:
        names = preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        return pd.DataFrame(columns=["Feature", "Importance"])
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(estimator.coef_), axis=0)
    else:
        return pd.DataFrame(columns=["Feature", "Importance"])
    return (
        pd.DataFrame({"Feature": names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )