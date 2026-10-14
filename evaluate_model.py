"""Audit the shipped model and run an exploratory unseen-restaurant evaluation."""
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sentiment import normalize_review

ROOT = Path(__file__).resolve().parent

def audit():
    model = joblib.load(ROOT / 'models/review_sentiment_model.joblib')
    data = pd.read_csv(ROOT / 'data/zomato_cleaned_step1.csv')
    data['normalized'] = data['Review'].fillna('').map(normalize_review)
    data = data[data['normalized'].str.len() > 0].reset_index(drop=True)
    train, test = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(data['Review'], data['Sentiment'], data['normalized']))
    predicted = model.predict(data.iloc[test]['Review'])
    errors = data.iloc[test][['Review', 'Sentiment']].copy()
    errors['Predicted sentiment'] = predicted
    errors = errors[errors['Sentiment'] != errors['Predicted sentiment']]
    errors['Review'] = errors['Review'].str.slice(0, 500)
    errors.head(100).to_csv(ROOT / 'data/model_error_examples.csv', index=False)
    # Freeze existing parameters; this is a diagnostic, not a new model-selection score.
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(data, groups=data['Restaurant']))
    test_groups = set(data.iloc[test]['normalized'])
    train = [i for i in train if data.iloc[i]['normalized'] not in test_groups]
    assert set(data.iloc[train]['Restaurant']).isdisjoint(data.iloc[test]['Restaurant'])
    assert set(data.iloc[train]['normalized']).isdisjoint(test_groups)
    diagnostic = clone(model).fit(data.iloc[train]['Review'], data.iloc[train]['Sentiment'])
    prediction = diagnostic.predict(data.iloc[test]['Review'])
    results = {'purpose': 'Exploratory evaluation; fixed existing hyperparameters, not nested model selection',
        'split': '20% of restaurants held out; exact normalized text overlap removed from training',
        'train_rows': len(train), 'test_rows': len(test),
        'held_out_restaurants': int(data.iloc[test]['Restaurant'].nunique()),
        'accuracy': accuracy_score(data.iloc[test]['Sentiment'], prediction),
        'f1_macro': f1_score(data.iloc[test]['Sentiment'], prediction, average='macro'),
        'classification_report': classification_report(data.iloc[test]['Sentiment'], prediction, output_dict=True, zero_division=0),
        'original_test_errors': len(errors),
        'neutral_test_errors': int(errors['Sentiment'].eq('Neutral').sum())}
    (ROOT / 'data/model_diagnostics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    audit()
