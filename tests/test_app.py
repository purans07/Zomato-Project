import unittest
import joblib
from streamlit.testing.v1 import AppTest
from sentiment import normalize_review
from train_model import ROOT

class SentimentAppTests(unittest.TestCase):
    def test_negation_and_noise(self):
        self.assertEqual(normalize_review('The food is NOT good!'), 'food not good')
        self.assertIn('not', normalize_review("I wouldn’t recommend it"))
        self.assertEqual(normalize_review('https://example.com 123 !!!'), '')

    def test_saved_pipeline_accepts_raw_text(self):
        model = joblib.load(ROOT / 'models/review_sentiment_model.joblib')
        reviews = ['Excellent food and friendly service', 'Worst food and terrible service', 'The meal was average']
        self.assertEqual(list(model.predict(reviews)), ['Positive', 'Negative', 'Neutral'])
        self.assertEqual(model.named_steps['tfidf'].preprocessor, normalize_review)

    def test_dashboard_and_prediction(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.text_area[0].input('Excellent food and friendly service')
        app.button(key='FormSubmitter:review_form-Analyze sentiment').click().run()
        self.assertFalse(app.exception)
        self.assertIn('Positive', app.success[0].value)
        app.text_area[0].input('123 !!!')
        app.button(key='FormSubmitter:review_form-Analyze sentiment').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.warning)
        app.multiselect[0].select('Chinese').run()
        self.assertFalse(app.exception)
        app.slider[0].set_value((1.0, 1.0)).run()
        self.assertFalse(app.exception)
        app.number_input[0].set_value(1000).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.info)

if __name__ == '__main__':
    unittest.main()
