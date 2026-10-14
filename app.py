"""Zomato review analytics dashboard. Run with streamlit run app.py."""
import json
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sentiment import normalize_review
from insights import explain_review, highlight_review, common_phrases, analyze_batch, safe_csv, MAX_BYTES, MAX_ROWS

ROOT = Path(__file__).resolve().parent
COLORS = {'Positive': '#238B68', 'Neutral': '#E6A23C', 'Negative': '#E23744'}
st.set_page_config(page_title='Zomato | Review Explorer', page_icon='🍽️', layout='wide', initial_sidebar_state='collapsed')
# Hide hosting/developer actions while keeping the sidebar toggle available.
st.markdown("""
<style>
[data-testid="stToolbar"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] {
    display: none !important;
}
[data-testid="stMainBlockContainer"] {max-width: 1400px; padding-top: 2.5rem;}
[data-testid="stMetric"] {background: #F8F4F3; border: 1px solid #E7DDDA; border-radius: 14px; padding: 16px;}
.hero {background: linear-gradient(120deg, #7C1625, #B42335); color: white; padding: 30px; border-radius: 20px; margin-bottom: 20px;}
.hero h1 {color: white; font-size: clamp(1.8rem, 4vw, 3rem); padding: 0 0 8px;}
.hero p {margin: 0; color: #FFF3F3;}
mark {background: #FBE7A1; color: #302500; padding: 2px 4px; border-radius: 4px;}
@media(max-width: 700px) {.hero {padding: 20px;} [data-testid="stMainBlockContainer"] {padding: 2.8rem 1rem 1rem;}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    frame = pd.read_csv(ROOT / 'data/zomato_cleaned_step1.csv')
    frame['Time'] = pd.to_datetime(frame['Time'], errors='coerce')
    return frame

@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'models/review_sentiment_model.joblib')

@st.cache_data
def load_results():
    return json.loads((ROOT / 'data/review_model_results.json').read_text())

st.markdown('<div class="hero"><h1>Zomato Review Explorer</h1><p>Discover great dining. Understand the feedback behind every rating.</p></div>', unsafe_allow_html=True)
try:
    data = load_data()
except (OSError, ValueError) as exc:
    st.error(f'The restaurant dataset could not be loaded: {exc}')
    st.stop()

def reset_filters():
    for key in ['cuisines', 'restaurants', 'ratings']:
        st.session_state.pop(key, None)

def choose_sample(text):
    st.session_state['review_text'] = text

with st.sidebar:
    st.button('Reset filters', on_click=reset_filters, use_container_width=True)
    st.header('Explore the dataset')
    cuisines = st.multiselect('Primary cuisine', sorted(data['Primary_Cuisine'].dropna().unique()), key='cuisines')
    choices = data if not cuisines else data[data['Primary_Cuisine'].isin(cuisines)]
    st.session_state['restaurants'] = [r for r in st.session_state.get('restaurants', []) if r in set(choices['Restaurant'])]
    restaurants = st.multiselect('Restaurant', sorted(choices['Restaurant'].dropna().unique()), key='restaurants')
    min_rating, max_rating = st.slider('Review rating', 1.0, 5.0, (1.0, 5.0), 0.5, key='ratings')
    st.caption('Filters apply to the overview and restaurant explorer.')
filtered = choices[choices['Rating_num'].between(min_rating, max_rating)]
if restaurants:
    filtered = filtered[filtered['Restaurant'].isin(restaurants)]

active = [*cuisines, *restaurants]
if (min_rating, max_rating) != (1.0, 5.0):
    active.append(f'Rating {min_rating:g}–{max_rating:g}')
st.caption(f"{len(filtered):,} of {len(data):,} reviews match • " + (' · '.join(active) if active else 'All cuisines and restaurants'))
overview, explorer, compare, predict, batch, evaluation = st.tabs(['Overview', 'Restaurants', 'Compare', 'Try a review', 'CSV analysis', 'Model & methodology'])
with overview:
    st.subheader('What do customers say?')
    if filtered.empty:
        st.info('No reviews match these filters. Try a wider selection.')
    else:
        a, b, c, d = st.columns(4)
        a.metric('Reviews', f'{len(filtered):,}')
        b.metric('Restaurants', filtered['Restaurant'].nunique())
        c.metric('Average rating', f"{filtered['Rating_num'].mean():.2f} / 5")
        d.metric('Positive reviews', f"{filtered['Sentiment'].eq('Positive').mean():.0%}")
        left, right = st.columns(2)
        counts = filtered['Sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Reviews')
        left.plotly_chart(px.pie(counts, names='Sentiment', values='Reviews', hole=.55,
            color='Sentiment', color_discrete_map=COLORS, title='Sentiment from review ratings'), use_container_width=True)
        right.plotly_chart(px.histogram(filtered, x='Rating_num', color_discrete_sequence=['#E23744'],
            title='Customer rating distribution', labels={'Rating_num': 'Rating', 'count': 'Reviews'}), use_container_width=True)
        monthly = filtered.dropna(subset=['Time']).assign(Month=lambda x: x['Time'].dt.to_period('M').astype(str)).groupby('Month').size().reset_index(name='Reviews')
        st.plotly_chart(px.line(monthly, x='Month', y='Reviews', title='Review activity over time', markers=True,
            color_discrete_sequence=['#E23744']), use_container_width=True)
        st.caption('Dataset dates: May 2016–May 2019. These charts show the provided sample, not current Zomato activity.')
with explorer:
    st.subheader('Explore restaurant rankings')
    minimum = st.number_input('Minimum reviews per restaurant', min_value=1, max_value=1000, value=20)
    summary = filtered.groupby('Restaurant').agg(Reviews=('Rating_num', 'size'), Rating=('Rating_num', 'mean'),
        Cost=('Cost_num', 'first'), Cuisine=('Primary_Cuisine', 'first')).reset_index()
    summary = summary[summary['Reviews'] >= minimum].sort_values('Rating', ascending=False)
    if summary.empty:
        st.info('No restaurants match. Reduce the minimum reviews or widen the filters.')
    else:
        st.plotly_chart(px.bar(summary.head(15).sort_values('Rating'), x='Rating', y='Restaurant', orientation='h',
            title='Highest-rated restaurants in this selection', range_x=[0, 5], color_discrete_sequence=['#E23744']), use_container_width=True)
        st.dataframe(summary, hide_index=True, width='stretch',
            column_config={'Rating': st.column_config.NumberColumn(format='%.2f'), 'Cost': st.column_config.NumberColumn('Approx. cost for two (₹)', format='₹%d')})
        st.download_button('Download restaurant summary', summary.to_csv(index=False), 'restaurant_summary.csv', 'text/csv')
with predict:
    st.subheader('Try your own restaurant review')
    st.write('Enter an English review to predict Positive, Neutral, or Negative sentiment.')
    st.caption('Start with an example or write your own review.')
    sample_cols = st.columns(3)
    for col, title, text in zip(sample_cols, ['Positive example', 'Neutral example', 'Negative example'],
        ['Excellent food and friendly service', 'The meal was average', 'Worst food and terrible service']):
        col.button(title, on_click=choose_sample, args=(text,), use_container_width=True)
    with st.form('review_form'):
        review = st.text_area('Your review', key='review_text', max_chars=5000, height=130, placeholder='How was the food, service, or ambience?')
        submitted = st.form_submit_button('Analyze sentiment', type='primary')
    if submitted:
        if not normalize_review(review):
            st.warning('Please enter a review containing meaningful words.')
        else:
            try:
                model = load_model()
                vector = model.named_steps['tfidf'].transform([review])
                if vector.nnz == 0:
                    st.warning('These words are outside the model vocabulary. Try a more descriptive English review.')
                else:
                    label = model.predict([review])[0]
                    st.success(f'Predicted sentiment: {label}')
                    probabilities = pd.DataFrame({'Sentiment': model.classes_, 'Model probability': model.predict_proba([review])[0]})
                    st.plotly_chart(px.bar(probabilities, x='Sentiment', y='Model probability', color='Sentiment',
                        color_discrete_map=COLORS, range_y=[0, 1]), use_container_width=True)
                    explanation = explain_review(model, review)
                    st.subheader('What influenced this prediction?')
                    st.caption('Highlighted words and phrases support the predicted class. Text below shows the cleaned words used by the model.')
                    st.markdown(highlight_review(review, explanation), unsafe_allow_html=True)
                    if not explanation.empty:
                        explanation['Effect'] = explanation['Contribution'].map(lambda x: 'Supports prediction' if x > 0 else 'Opposes prediction')
                        st.plotly_chart(px.bar(explanation.sort_values('Contribution'), x='Contribution', y='Phrase',
                            color='Effect', orientation='h', color_discrete_map={'Supports prediction': '#176B50', 'Opposes prediction': '#B42335'},
                            title='TF-IDF × coefficient for the predicted class'), use_container_width=True)
                    st.caption('Contributions explain this model’s score, not causation; the intercept and other classes also affect the result.')
                    st.caption('Probabilities are model estimates, not guaranteed confidence. Mixed opinions, sarcasm, and unfamiliar language may be misclassified.')
            except (OSError, ValueError, ImportError) as exc:
                st.error(f'The prediction model is unavailable. Please check the deployment: {exc}')
with compare:
    st.subheader('Two restaurants. One clear comparison.')
    st.caption('Uses all historical reviews for each restaurant, independently of sidebar filters.')
    names = sorted(data['Restaurant'].dropna().unique())
    if len(names) < 2:
        st.info('Two restaurants are needed for comparison.')
    else:
        pick_a, pick_b = st.columns(2)
        name_a = pick_a.selectbox('First restaurant', names)
        others = [name for name in names if name != name_a]
        name_b = pick_b.selectbox('Second restaurant', others)
        for col, name in zip(st.columns(2), [name_a, name_b]):
            with col:
                subset = data[data['Restaurant'] == name]
                st.markdown(f'### {name}')
                st.metric('Average rating', f"{subset['Rating_num'].mean():.2f} / 5")
                st.metric('Reviews', f'{len(subset):,}')
                st.metric('Positive reviews', f"{subset['Sentiment'].eq('Positive').mean():.0%}")
                costs = subset['Cost_num'].dropna()
                st.metric('Approx. cost for two', f'₹{costs.iloc[0]:,.0f}' if len(costs) else 'Unavailable')
                st.caption('Primary cuisine: ' + str(subset['Primary_Cuisine'].iloc[0]))
                for sentiment in ['Positive', 'Negative']:
                    st.markdown(f'**Common phrases in {sentiment.lower()} reviews**')
                    phrases = common_phrases(subset.loc[subset['Sentiment'] == sentiment, 'Review'])
                    if phrases.empty:
                        st.caption('Not enough repeated phrases in this group.')
                    else:
                        st.dataframe(phrases, hide_index=True, width='stretch')
        st.caption('Phrase counts describe recurring two-word phrases, not automatically identified causes. Different review volumes affect comparisons.')

with batch:
    st.subheader('Analyze a collection of reviews')
    st.write('Upload a CSV, select the review column, and download predictions with a status for every row.')
    st.caption(f'Limits: 5 MB, {MAX_ROWS:,} rows, 5,000 characters per review. Uploads are processed in memory and are not written to project files.')
    st.download_button('Download example CSV', b'Review\nExcellent food and friendly service\nThe meal was average\nWorst food and terrible service\n', 'example_reviews.csv', 'text/csv')
    uploaded = st.file_uploader('Review CSV', type=['csv'])
    if uploaded is not None:
        if uploaded.size > MAX_BYTES:
            st.warning('The file is too large. Please use a CSV under 5 MB.')
        else:
            try:
                from io import BytesIO
                frame = pd.read_csv(BytesIO(uploaded.getvalue()), nrows=MAX_ROWS + 1)
                if len(frame) > MAX_ROWS:
                    st.warning(f'Please upload at most {MAX_ROWS:,} rows.')
                elif frame.empty:
                    st.warning('The CSV contains no data rows.')
                else:
                    column = st.selectbox('Column containing review text', frame.columns,
                        index=list(frame.columns).index('Review') if 'Review' in frame else 0)
                    st.dataframe(frame.head(5), hide_index=True, width='stretch')
                    if st.button('Analyze CSV', type='primary'):
                        with st.spinner('Analyzing reviews…'):
                            output = analyze_batch(load_model(), frame, column)
                        analyzed = output['Analysis status'].eq('Analyzed')
                        st.success(f'{analyzed.sum():,} reviews analyzed; {(~analyzed).sum():,} rows skipped with a reason.')
                        counts = output.loc[analyzed, 'Predicted sentiment'].value_counts().rename_axis('Sentiment').reset_index(name='Reviews')
                        if not counts.empty:
                            st.plotly_chart(px.bar(counts, x='Sentiment', y='Reviews', color='Sentiment', color_discrete_map=COLORS), use_container_width=True)
                        st.dataframe(output, hide_index=True, width='stretch')
                        st.download_button('Download analyzed reviews', safe_csv(output), 'review_predictions.csv', 'text/csv', on_click='ignore')
                        st.caption('Model probabilities are uncalibrated. Empty and unknown-vocabulary reviews are skipped, not assigned a default sentiment.')
            except (ValueError, OSError, ImportError, UnicodeError) as exc:
                st.warning(f'Unable to analyze this CSV: {exc}')

with evaluation:
    st.subheader('How the model works')
    st.write('English review → shared text cleaning (negation preserved) → TF-IDF word and phrase features → Logistic Regression.')
    st.write('Labels come from customer ratings: 1–2 = Negative, above 2–3.5 = Neutral, above 3.5 = Positive. These are rating-derived labels, not independently annotated sentiment.')
    try:
        results = load_results()
        a, b, c = st.columns(3)
        a.metric('Held-out accuracy', f"{results['accuracy']:.2%}")
        b.metric('Macro F1', f"{results['f1_macro']:.2%}")
        c.metric('Held-out reviews', f"{results['test_rows']:,}")
        st.caption('Metrics below belong to the app model. The notebooks retain their original experiment results.')
        report = pd.DataFrame(results['classification_report']).T.loc[results['labels'], ['precision', 'recall', 'f1-score', 'support']]
        st.dataframe(report, width='stretch')
        st.plotly_chart(px.imshow(results['confusion_matrix'], x=results['labels'], y=results['labels'],
            text_auto=True, color_continuous_scale='Reds', labels={'x': 'Predicted', 'y': 'Actual'}, title='Held-out confusion matrix'), use_container_width=True)
        st.write(f"Training reviews: {results['train_rows']:,}. Identical normalized reviews are kept in the same split, including during cross-validation. Parameters are chosen using training-fold macro F1.")
        st.write('Limitations: historical restaurant sample; class imbalance; rating-derived labels; limited English vocabulary. Similar reviews and restaurant-specific patterns may still limit generalization.')
        with st.expander('Where the model struggles'):
            audit_path = ROOT / 'data/model_diagnostics.json'
            if audit_path.exists():
                audit = json.loads(audit_path.read_text())
                st.write(f"The original held-out test contains {audit['original_test_errors']:,} mistakes, including {audit['neutral_test_errors']:,} reviews labeled Neutral by their ratings.")
                st.write('Mixed feedback and rating/text disagreement are useful cases for further annotation and model development.')
                examples = pd.read_csv(ROOT / 'data/model_error_examples.csv')
                actual = st.selectbox('Actual rating-derived label', ['All', 'Negative', 'Neutral', 'Positive'])
                if actual != 'All':
                    examples = examples[examples['Sentiment'] == actual]
                st.dataframe(examples, hide_index=True, width='stretch')
                st.caption('Up to 100 misclassified examples from the original held-out split; long reviews are shortened.')
                st.markdown('**Exploratory test on unseen restaurants**')
                st.write(f"A separate model with the same parameters was evaluated on {audit['held_out_restaurants']} restaurants excluded from its training. Accuracy: {audit['accuracy']:.2%}; macro F1: {audit['f1_macro']:.2%}.")
                st.caption('Exact normalized review overlap was removed from training. This is a diagnostic with fixed existing parameters, not a fresh nested model-selection result. It does not replace the shipped model or its primary metrics.')
        with st.expander('Original notebook model comparison'):
            original = json.loads((ROOT / 'data/ml_model_results.json').read_text())
            st.dataframe(pd.DataFrame([{'Model': name, 'Accuracy': item['tuned_metrics']['accuracy'], 'Macro F1': item['tuned_metrics']['f1_macro']} for name, item in original['models'].items()]), hide_index=True)
            st.caption('Historical experiments used different preprocessing and a random row split; results are not directly comparable to the app evaluation.')
    except (OSError, ValueError) as exc:
        st.error(f'Model evaluation results could not be loaded: {exc}')
