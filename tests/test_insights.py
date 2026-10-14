import unittest
import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest
from insights import analyze_batch, explain_review, highlight_review, common_phrases, safe_csv
from train_model import ROOT

class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load(ROOT / 'models/review_sentiment_model.joblib')

    def test_batch_preserves_rows_and_flags_invalid(self):
        frame = pd.DataFrame({'Review': ['Excellent food and friendly service', None, '123 !!!', 'zzxyqzzzz']})
        out = analyze_batch(self.model, frame, 'Review')
        self.assertEqual(len(out), 4)
        self.assertEqual(out.iloc[0]['Predicted sentiment'], 'Positive')
        self.assertEqual(list(out['Analysis status'])[1:], ['Empty or no meaningful words', 'Empty or no meaningful words', 'Outside model vocabulary'])
        with self.assertRaises(ValueError):
            analyze_batch(self.model, pd.DataFrame({'Review': ['x'] * 2001}), 'Review')
        with self.assertRaises(ValueError):
            analyze_batch(self.model, pd.DataFrame({'Review': ['x' * 5001]}), 'Review')
        with self.assertRaises(ValueError):
            analyze_batch(self.model, out, 'Review')

    def test_explanation_matches_model_score(self):
        text = 'Excellent food and friendly service'
        explanation = explain_review(self.model, text, limit=10000)
        clf = self.model.named_steps['classifier']
        index = list(clf.classes_).index(self.model.predict([text])[0])
        score = self.model.decision_function([text])[0, index]
        self.assertAlmostEqual(explanation.Contribution.sum() + clf.intercept_[index], score)
        self.assertIn('<mark>', highlight_review(text, explanation))
        self.assertNotIn('<script>', highlight_review('<script>alert(1)</script>', explanation))

    def test_phrase_document_counts_and_safe_export(self):
        phrases = common_phrases(['Great service great service', 'Great service'])
        self.assertEqual(phrases.iloc[0]['Reviews mentioning phrase'], 2)
        self.assertTrue(common_phrases(['hello']).empty)
        self.assertIn("'=SUM", safe_csv(pd.DataFrame({'Review': ['=SUM(1)']})).decode('utf-8-sig'))

    def test_sample_filters_and_comparison(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == 'Neutral example').click().run()
        self.assertEqual(app.text_area[0].value, 'The meal was average')
        app.button(key='FormSubmitter:review_form-Analyze sentiment').click().run()
        self.assertFalse(app.exception)
        self.assertIn('Neutral', app.success[0].value)
        app.multiselect(key='cuisines').select('Chinese').run()
        next(b for b in app.button if b.label == 'Reset filters').click().run()
        self.assertEqual(app.multiselect(key='cuisines').value, [])
        self.assertEqual(app.slider(key='ratings').value, (1.0, 5.0))
        app.selectbox[0].select_index(3).run()
        self.assertFalse(app.exception)
        self.assertNotEqual(app.selectbox[0].value, app.selectbox[1].value)

if __name__ == '__main__':
    unittest.main()
