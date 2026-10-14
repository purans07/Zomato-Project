# Zomato Review Explorer

**Live app:** https://zomato-review-explorer.streamlit.app/

An interactive restaurant review analytics application: explore historical restaurant reviews, compare restaurants, and classify an English review as Positive, Neutral, or Negative.

## App features

- Overview with rating distribution, sentiment breakdown, and monthly review activity.
- Restaurant and primary-cuisine filters, restaurant rankings, and CSV summary download.
- Live review classification using TF-IDF and Logistic Regression.
- Held-out evaluation, per-class metrics, confusion matrix, and methodology.

The dataset is historical (May 2016–May 2019). This independent project is not affiliated with Zomato and does not access live Zomato data.

## New interactive features

- Responsive branded header, metric cards, filter summaries, matching review counts and one-click filter reset.
- Clickable Positive/Neutral/Negative examples and signed feature contributions for predictions, with supporting words highlighted in cleaned text.
- Two-restaurant comparison including cost, ratings, sentiment share and common positive/negative review phrases.
- CSV review analysis with a selectable column, downloadable results, per-row validation, a 5 MB/2,000-row limit and spreadsheet-safe exports.
- Error examples and an exploratory evaluation on restaurants excluded from training.

Comparison uses all reviews for the selected restaurants, independently of sidebar filters. Phrase counts are document counts of repeated bigrams, not causal explanations. Uploaded CSV data is processed in memory without being saved to project files.

To regenerate diagnostics after retraining, run `python evaluate_model.py`. The diagnostic fits a separate estimator and does not replace the shipped artifact. The existing primary evaluation is preserved.

## Run locally

Use **Python 3.12** to match the tested deployment environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app uses the included model; training is not performed during startup. No API key or database is required.

## Deploy on Streamlit Community Cloud

1. Commit and push the app files, `sentiment.py`, `requirements.txt`, `.streamlit/config.toml`, `insights.py`, `data/model_diagnostics.json`, `data/model_error_examples.csv`, `models/review_sentiment_model.joblib` and `data/review_model_results.json`, and existing CSV/JSON data to GitHub.
2. Sign in to Streamlit Community Cloud and connect GitHub.
3. Create an app from `purans07/Zomato-Project`, selecting the branch containing the app.
4. Set the entrypoint to `app.py` and choose **Python 3.12** in advanced settings.
5. Deploy, try the filters and review form, then update the live application URL and screenshot in this README.

Deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

## Model and reproducibility

```bash
python train_model.py
python -m unittest discover -s tests -v
```

The app artifact is `models/review_sentiment_model.joblib`, with results in `data/review_model_results.json`. The complete pipeline accepts raw text and applies the same `sentiment.normalize_review` function during training and prediction. It expands contractions, removes URLs/digits/punctuation, and preserves negation. No extra language corpus download is needed.

A five-fold stratified group split reserves one fold for testing. Identical normalized reviews remain together in both the held-out split and three-fold training cross-validation. Grid search selects Logistic Regression parameters by macro F1. The saved estimator is fitted only on the training partition, matching the reported held-out evaluation.

Original notebook experiments remain in `models/zomato_sentiment_model.joblib` and `data/ml_model_results.json`. Their preprocessing and split differ, so their scores are not directly comparable to the app model.

## App model results

- Held-out accuracy: **82.85%**
- Macro F1: **72.61%**
- Neutral-class F1: **43.51%**
- Training reviews: **7,847**; held-out reviews: **2,082**

These scores use the duplicate-aware split described above. They should not be read as a direct improvement over the original notebook because the evaluation split changed.

## Limitations

- Labels are derived from ratings: 1–2 Negative; above 2–3.5 Neutral; above 3.5 Positive. They are not independently annotated sentiment labels.
- The historical sample and class imbalance limit generalization. Neutral, mixed, sarcastic, and non-English reviews can be misclassified.
- Duplicate grouping prevents identical normalized text from crossing splits, but does not eliminate near-duplicates or restaurant-specific patterns.
- Predicted probabilities are uncalibrated model estimates, not guaranteed confidence.
- Charts describe the supplied sample, not the whole Zomato platform.

## Original analysis

The notebooks cover cleaning, exploratory analysis, statistical comparisons, and experiments with Naive Bayes, Logistic Regression, and Linear SVM. Install `requirements-notebook.txt` to run them in Jupyter. Notebook results are historical; approximate statistical tests and heuristic NLP demonstrations should be considered when interpreting them.

## Data attribution

Inputs are `Zomato Restaurant reviews.csv` and `Zomato Restaurant names and Metadata.csv`, supplied with the original capstone project. The upstream dataset URL and redistribution license have not been documented in this repository; add verified attribution and terms before redistributing the data beyond their permitted use. No ownership of Zomato's brand or source data is claimed.
