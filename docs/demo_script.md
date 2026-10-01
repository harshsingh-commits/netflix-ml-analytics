# Demo video script (about 4 minutes)

## 0:00–0:25 — Opening

“This is Netflix Catalog Intelligence, a Streamlit application for exploring catalog composition, rating categories, content segments, and observed catalog-add trends. The results are generated from the CSV loaded for this demonstration.”

## 0:25–0:55 — Load data and dashboard

“I’ll upload the Netflix titles CSV using the sidebar. The app normalizes the common Netflix schema, removes duplicate title records, and derives useful fields such as genre count, country, duration, cast count, and director count. The dashboard summarizes the records that match the current filters.”

Show the title, format, genre, and release-year charts. Briefly apply a genre or content-type filter, then clear it.

## 0:55–1:35 — Rating classification

“The target is the dataset’s existing content-rating category, not audience satisfaction or popularity. I’ll train the classifiers using a stratified holdout, compare accuracy and weighted precision, recall, and F1, then inspect the confusion matrix and feature-importance ranking. Random Forest GridSearchCV is optional and runs only on the training partition.”

Click **Train and compare models** and show the measured model chart and confusion matrix.

## 1:35–2:15 — Segmentation and recommendations

“Segmentation combines numeric metadata with one-hot encoded format, country, and leading genres. The app compares cluster counts with inertia and silhouette score, then displays PCA or t-SNE projections. These are descriptive groups, not official Netflix categories.”

Run segmentation. Show a cluster profile and select one title to see nearest neighbors within its segment.

## 2:15–3:00 — Forecasting

“Forecasting uses actual `date_added` months to model catalog additions. It compares Random Forest autoregression, ARIMA, and Prophet on a time-based holdout and reports RMSE, MAE, and R². The selected view forecasts the next 12 or 24 months. Original release-year counts are separate.”

Run the forecast and show the measured chart and model table. If date coverage is insufficient, explain the app’s displayed requirement instead of implying it produced a forecast.

## 3:00–3:40 — Analytics and exports

“The analytics engine combines missingness, correlation, genre, country, rating, and catalog-add analysis into a compact set of insights. When a supported classification model is available, SHAP can add a local model explanation. I can export a PDF, a ZIP of CSV tables, and a PNG chart.”

Show the insights and report controls; download one report if time permits.

## 3:40–4:00 — Close

“The key caveat is that catalog metadata can describe content patterns, but it does not measure viewing success. Forecasts extend observed additions rather than predict official company plans. This project provides a reproducible starting point for deeper analysis when richer, licensed outcome data is available.”