"""Rebuild the app model with a duplicate-aware held-out evaluation."""
import json
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sentiment import normalize_review

ROOT = Path(__file__).resolve().parent
LABELS = ['Negative', 'Neutral', 'Positive']

def train():
    data = pd.read_csv(ROOT / 'data/zomato_cleaned_step1.csv')
    data['normalized'] = data['Review'].fillna('').map(normalize_review)
    data = data[data['normalized'].str.len() > 0].reset_index(drop=True)
    X, y, groups = data['Review'], data['Sentiment'], data['normalized']
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    assert set(groups.iloc[train_idx]).isdisjoint(groups.iloc[test_idx])
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(preprocessor=normalize_review, lowercase=False,
                                max_features=5000, ngram_range=(1, 2), min_df=2)),
        ('classifier', LogisticRegression(max_iter=1000, solver='lbfgs')),
    ])
    search = GridSearchCV(pipeline, {'classifier__C': [0.5, 1.0, 2.0],
        'classifier__class_weight': [None, 'balanced']}, scoring='f1_macro',
        cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=43), n_jobs=1)
    search.fit(X.iloc[train_idx], y.iloc[train_idx], groups=groups.iloc[train_idx])
    prediction = search.predict(X.iloc[test_idx])
    results = {
        'model': 'Logistic Regression', 'sklearn_version': sklearn.__version__,
        'train_rows': len(train_idx), 'test_rows': len(test_idx),
        'split': 'StratifiedGroupKFold: identical normalized reviews kept together',
        'accuracy': accuracy_score(y.iloc[test_idx], prediction),
        'f1_macro': f1_score(y.iloc[test_idx], prediction, average='macro'),
        'best_cv_f1_macro': search.best_score_, 'best_params': search.best_params_,
        'classification_report': classification_report(y.iloc[test_idx], prediction, output_dict=True, zero_division=0),
        'labels': LABELS, 'confusion_matrix': confusion_matrix(y.iloc[test_idx], prediction, labels=LABELS).tolist(),
        'target_distribution': y.value_counts().to_dict(),
        'preprocessing': 'Shared raw-text pipeline; negation preserved; no stemming',
    }
    # Preserve the notebook's original artifact and historical metrics.
    joblib.dump(search.best_estimator_, ROOT / 'models/review_sentiment_model.joblib')
    (ROOT / 'data/review_model_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    train()
