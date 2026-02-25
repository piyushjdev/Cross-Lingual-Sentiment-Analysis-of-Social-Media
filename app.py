import os
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import pandas as pd
import altair as alt
from googletrans import Translator

# Silence TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TRANSFORMERS_NO_TF'] = '1'

st.set_page_config(page_title="Sentiment Analysis App", page_icon="📝")
st.title("📝 Sentiment Analysis App")
st.write("Analyze sentiment of text, social media posts, and product reviews with multi-language support.")

# Load Model
@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Translator
translator = Translator()

# Sidebar options
option = st.sidebar.selectbox(
    "Choose Input Type",
    ["Text", "CSV Reviews", "Social Media (Twitter Handle/Hashtag)"]
)

# Function to analyze sentiment
def analyze_sentiment(text):
    # Translate if not English
    detected_lang = translator.detect(text).lang
    if detected_lang != 'en':
        text = translator.translate(text, dest='en').text

    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded)
    logits = output.logits.detach().cpu().numpy()[0]
    scores = softmax(logits)
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment_label = labels[scores.argmax()]
    sentiment_score = scores[scores.argmax()]
    return sentiment_label, sentiment_score, scores

# ---------------- Text Input ----------------
if option == "Text":
    tweet = st.text_area("Enter text:", "I am going")
    if st.button("Analyze"):
        if tweet.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            label, score, scores = analyze_sentiment(tweet)
            st.success(f"*Sentiment:* {label} ({score:.2%})")
            
            # Bar chart
            df = pd.DataFrame({'Sentiment': ['Negative', 'Neutral', 'Positive'], 'Score': scores})
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Score', axis=alt.Axis(format='%')),
                y=alt.Y('Sentiment', sort=['Negative', 'Neutral', 'Positive']),
                color='Sentiment'
            ).properties(width=500, height=200)
            st.altair_chart(chart)

# ---------------- CSV Reviews ----------------
elif option == "CSV Reviews":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("CSV must have a 'review' column")
        else:
            sentiments = df['review'].apply(lambda x: analyze_sentiment(str(x))[0])
            df['Sentiment'] = sentiments
            st.write(df.head())

            # Summary chart
            summary = df['Sentiment'].value_counts().reset_index()
            summary.columns = ['Sentiment', 'Count']
            chart = alt.Chart(summary).mark_bar().encode(
                x='Count', 
                y=alt.Y('Sentiment', sort=['Negative', 'Neutral', 'Positive']),
                color='Sentiment'
            ).properties(width=500, height=300)
            st.altair_chart(chart)

# ---------------- Social Media (Twitter) ----------------
elif option == "Social Media (Twitter Handle/Hashtag)":
    st.info("⚠️ Twitter API setup required for fetching posts in real-time.")
    query = st.text_input("Enter Twitter handle or hashtag (e.g., @username or #topic):")
    if st.button("Analyze Tweets"):
        if query.strip() == "":
            st.warning("Enter a valid handle or hashtag.")
        else:
            st.info("Fetching tweets... (Requires Twitter API credentials)")
            st.warning("Twitter integration not coded here — implement using Tweepy or Twitter API v2.")
            st.success("Once implemented, this will fetch recent tweets and analyze their sentiment.")

st.markdown("---")
