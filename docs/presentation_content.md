# PowerPoint content: Netflix catalog intelligence

## Slide 1 — Problem statement

- A content catalog has many formats, ratings, genres, release periods, and country credits.
- The project turns a Netflix titles CSV into an interactive, reproducible analysis workflow.
- Scope: describe catalog structure, predict the existing rating category, segment titles, and forecast observed catalog additions.

## Slide 2 — Dataset overview

- Source: the Netflix CSV supplied for the demonstration.
- Show the actual row count, date coverage, missing-value profile, and Movie/TV Show split from the Dashboard after upload.
- Explain the key fields: `type`, `rating`, `release_year`, `duration`, `listed_in`, `country`, and `date_added`.
- Distinguish original release year from date added to Netflix.

## Slide 3 — Methodology

- Normalize columns, remove duplicates, handle missing values, and parse dates and durations.
- Derive director/cast/genre counts, primary country, duration value/unit, release decade, and catalog-add month/year.
- Use one shared filtered dataset so the dashboard and task pages stay consistent.
- Use a stratified holdout for rating classification and a temporal holdout for forecasts.

## Slide 4 — Models used

- Classification: Logistic Regression, Random Forest, Decision Tree, and XGBoost; GridSearchCV tunes Random Forest.
- Segmentation: KMeans and Agglomerative clustering; silhouette score selects K and PCA/t-SNE visualize structure.
- Forecasting: Random Forest autoregression, ARIMA(1,1,1), and Prophet when dependencies and monthly dates are available.
- Explainability: model feature importance and optional SHAP values.

## Slide 5 — Results

- Present the live model-comparison and confusion-matrix charts; report measured accuracy, weighted precision, recall, and F1 from the uploaded data.
- Show elbow/silhouette evidence, cluster profiles, and representative nearest-title recommendations.
- Show the chosen forecast horizon with the temporal holdout RMSE, MAE, and R².
- Do not copy example or synthetic values: read the dashboard outputs generated for the demonstration CSV.

## Slide 6 — Automated insights and reporting

- Highlight the measured top genres, country credits, rating mix, release/addition trends, and fastest-growing genres.
- Demonstrate the PDF report, CSV bundle, and PNG chart export.
- Explain that “fastest-growing” compares observed catalog-add periods and is not a causal or popularity claim.

## Slide 7 — Conclusion and next steps

- The application creates a repeatable catalog-analysis baseline from a single uploaded CSV.
- Forecasts describe historical addition patterns and should not be presented as company guidance.
- Next steps: validate on updated data, review class imbalance and drift, and add licensed audience/viewership outcomes before making popularity claims.