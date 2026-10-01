"""Time-series forecasting of observed Netflix catalog additions."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import RANDOM_STATE

LOGGER = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Monthly history, annual catalog trend, forecasts, and holdout scores."""

    monthly_history: pd.Series
    annual_releases: pd.Series
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    insights: list[str]
    unavailable_models: list[str]


def create_release_series(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Aggregate actual catalog-addition dates by month and release years by year."""
    annual = pd.to_numeric(frame["release_year"], errors="coerce").dropna().astype(int).value_counts().sort_index()
    annual.index.name = "release_year"
    if "date_added_dt" not in frame or frame["date_added_dt"].notna().sum() == 0:
        return pd.Series(dtype=float, name="titles_added"), annual.astype(int)
    months = frame["date_added_dt"].dropna().dt.to_period("M").value_counts().sort_index()
    monthly = months.asfreq("M", fill_value=0)
    monthly.index = monthly.index.to_timestamp()
    monthly = monthly.astype(float)
    monthly.name = "titles_added"
    return monthly, annual.astype(int)


def build_lag_features(series: pd.Series) -> pd.DataFrame:
    """Create autoregressive lag, rolling mean, and trend-index features."""
    frame = pd.DataFrame({"target": series.astype(float)})
    for lag in (1, 2, 3, 6, 12):
        frame[f"lag_{lag}"] = frame["target"].shift(lag)
    frame["rolling_mean_3"] = frame["target"].shift(1).rolling(3).mean()
    frame["trend"] = np.arange(len(frame), dtype=float)
    return frame.dropna()


def _recursive_random_forest(history: np.ndarray, horizon: int) -> np.ndarray:
    feature_frame = build_lag_features(pd.Series(history))
    if feature_frame.empty:
        raise ValueError("At least 13 monthly observations are required for lag features.")
    columns = [column for column in feature_frame if column != "target"]
    model = RandomForestRegressor(
        n_estimators=240,
        min_samples_leaf=2,
        max_features=0.9,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(feature_frame[columns], feature_frame["target"])
    values = list(history.astype(float))
    predictions: list[float] = []
    for _ in range(horizon):
        position = len(values)
        row = {
            "lag_1": values[-1],
            "lag_2": values[-2],
            "lag_3": values[-3],
            "lag_6": values[-6],
            "lag_12": values[-12],
            "rolling_mean_3": float(np.mean(values[-3:])),
            "trend": float(position),
        }
        prediction = max(0.0, float(model.predict(pd.DataFrame([row], columns=columns))[0]))
        predictions.append(prediction)
        values.append(prediction)
    return np.asarray(predictions)


def _arima_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(history, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
        return np.maximum(0.0, np.asarray(fitted.forecast(steps=horizon), dtype=float))


def _prophet_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    from prophet import Prophet

    training = pd.DataFrame({"ds": series.index, "y": series.to_numpy(dtype=float)})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(training)
        future = model.make_future_dataframe(periods=horizon, freq="MS", include_history=False)
        return np.maximum(0.0, model.predict(future)["yhat"].to_numpy(dtype=float))


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted, force_finite=True)),
    }


def forecast_releases(frame: pd.DataFrame) -> ForecastResult:
    """Evaluate candidate models and produce 12- and 24-month forecasts."""
    monthly, annual = create_release_series(frame)
    if len(monthly) < 30:
        raise ValueError(
            "At least 30 distinct calendar months with date_added values are needed for a monthly forecast. "
            "The annual release-year trend is still available."
        )
    test_size = min(12, max(3, len(monthly) // 5))
    train_series = monthly.iloc[:-test_size]
    actual = monthly.iloc[-test_size:].to_numpy(dtype=float)
    evaluator = {
        "Random forest": lambda series, horizon: _recursive_random_forest(series.to_numpy(dtype=float), horizon),
        "ARIMA (1,1,1)": lambda series, horizon: _arima_forecast(series.to_numpy(dtype=float), horizon),
        "Prophet": lambda series, horizon: _prophet_forecast(series, horizon),
    }
    scores: list[dict[str, float | str]] = []
    forecasts: list[pd.DataFrame] = []
    unavailable: list[str] = []
    for name, forecaster in evaluator.items():
        try:
            validation_prediction = forecaster(train_series, test_size)
            scores.append({"Model": name, **_score(actual, validation_prediction)})
            for horizon in (12, 24):
                values = forecaster(monthly, horizon)
                dates = pd.date_range(monthly.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
                forecasts.append(pd.DataFrame({"Date": dates, "Forecast": values, "Model": name, "Horizon": horizon}))
        except (ImportError, ModuleNotFoundError) as exc:
            LOGGER.info("Forecast model %s is unavailable: %s", name, exc)
            unavailable.append(f"{name}: dependency is not installed")
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            LOGGER.warning("Forecast model %s could not be fitted: %s", name, exc)
            unavailable.append(f"{name}: {exc.__class__.__name__}")
    if not forecasts:
        raise RuntimeError("No forecasting model completed successfully. Check statsmodels and the input time series.")
    forecast_frame = pd.concat(forecasts, ignore_index=True)
    metric_frame = pd.DataFrame(scores).sort_values("RMSE").reset_index(drop=True)
    insights = _forecast_insights(monthly, forecast_frame)
    return ForecastResult(
        monthly_history=monthly,
        annual_releases=annual,
        forecasts=forecast_frame,
        metrics=metric_frame,
        insights=insights,
        unavailable_models=unavailable,
    )


def _forecast_insights(history: pd.Series, forecasts: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    annualized_history = float(history.tail(12).sum())
    twelve_months = forecasts[forecasts["Horizon"] == 12]
    for model, group in twelve_months.groupby("Model"):
        predicted = float(group["Forecast"].sum())
        change = (predicted / annualized_history - 1) * 100 if annualized_history else 0.0
        insights.append(
            f"{model} projects approximately {predicted:,.0f} additions over the next 12 months, "
            f"a {change:+.1f}% change versus the last 12 observed months."
        )
    if len(history) >= 24:
        prior = float(history.iloc[-24:-12].sum())
        recent = float(history.iloc[-12:].sum())
        growth = (recent / prior - 1) * 100 if prior else 0.0
        insights.append(f"Observed additions in the latest 12 months changed {growth:+.1f}% from the preceding year.")
    return insights