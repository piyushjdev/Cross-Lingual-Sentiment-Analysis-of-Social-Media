import os
import streamlit as st
import pandas as pd
import altair as alt
from transformers import pipeline
from googletrans import Translator

# Silence warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TRANSFORMERS_NO_TF"] = "1"

st.set_page_config(page_title="Sentiment Analysis App", page_icon="📝")

st.title("📝 Sentiment Analysis App")
st.write(
    "Analyze sentiment of text and product reviews with multi-language support."
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

sentiment_pipeline = load_model()

translator = Translator()

# ---------------- SENTIMENT FUNCTION ----------------
def analyze_sentiment(text):
    # Detect language
    detected_lang = translator.detect(text).lang
    if detected_lang != "en":
        text = translator.translate(text, dest="en").text

    result = sentiment_pipeline(text)[0]

    raw_label = result["label"]
    score = result["score"]

    # Convert LABEL_X to readable format
    label_map = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive"
    }

    sentiment_label = label_map.get(raw_label, raw_label)

    scores = [0, 0, 0]
    if sentiment_label == "Negative":
        scores[0] = score
    elif sentiment_label == "Neutral":
        scores[1] = score
    else:
        scores[2] = score

    return sentiment_label, score, scores


# ---------------- SIDEBAR ----------------
option = st.sidebar.selectbox(
    "Choose Input Type",
    ["Text", "CSV Reviews"]
)

# ---------------- TEXT INPUT ----------------
if option == "Text":
    user_text = st.text_area("Enter text:", "")

    if st.button("Analyze"):
        if user_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                label, score, scores = analyze_sentiment(user_text)

            st.success(f"Sentiment: **{label}** ({score:.2%})")

            df = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "Score": scores
            })

            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("Score", axis=alt.Axis(format="%")),
                y=alt.Y("Sentiment", sort=["Negative", "Neutral", "Positive"]),
                color="Sentiment"
            ).properties(width=500, height=300)

            st.altair_chart(chart, use_container_width=True)


# ---------------- CSV INPUT ----------------
elif option == "CSV Reviews":
    uploaded_file = st.file_uploader("Upload CSV (must contain 'review' column)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            with st.spinner("Analyzing reviews..."):
                sentiments = df["review"].apply(
                    lambda x: analyze_sentiment(str(x))[0]
                )

            df["Sentiment"] = sentiments
            st.write(df.head())

            summary = df["Sentiment"].value_counts().reset_index()
            summary.columns = ["Sentiment", "Count"]

            chart = alt.Chart(summary).mark_bar().encode(
                x="Count",
                y=alt.Y("Sentiment", sort=["Negative", "Neutral", "Positive"]),
                color="Sentiment"
            ).properties(width=500, height=300)

            st.altair_chart(chart, use_container_width=True)

st.markdown("---")
st.caption("Built with Streamlit, Transformers & Google Translate")
