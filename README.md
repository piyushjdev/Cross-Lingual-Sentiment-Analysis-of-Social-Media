# 📝 Cross-Lingual Sentiment Analysis of Social Media

## 📌 Overview
This project is a multilingual sentiment analysis web application built using HuggingFace Transformers and Streamlit.  
It analyzes text, product reviews, and social media content using a domain-adapted RoBERTa model.

The system automatically detects language and translates non-English text to English before performing sentiment classification.

---

## 🚀 Features

- 🌍 Multi-language sentiment analysis
- 🤖 Pretrained RoBERTa (Twitter sentiment model)
- 📂 CSV bulk review analysis
- 📊 Interactive sentiment visualization (Altair)
- ⚡ Fast model loading using caching
- 🖥 Streamlit web interface

---

## 🛠 Tech Stack

- Python
- Streamlit
- HuggingFace Transformers
- PyTorch
- Pandas
- Altair
- SciPy
- Google Translate API

---

## 🧠 Model Used

**cardiffnlp/twitter-roberta-base-sentiment-latest**

This model is fine-tuned for social media sentiment classification and predicts:

- Negative
- Neutral
- Positive

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/piyushjdev/Cross-Lingual-Sentiment-Analysis-of-Social-Media.git
cd Cross-Lingual-Sentiment-Analysis-of-Social-Media
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📊 Use Cases

- Social media monitoring
- Product review analysis
- Brand sentiment tracking
- Multilingual NLP research
- Real-time sentiment dashboards

---

## 🔮 Future Improvements

- Twitter API v2 integration
- Model fine-tuning on custom datasets
- REST API version using FastAPI
- Deployment on AWS / Docker
- Improved translation reliability

---

## 👨‍💻 Author

**Piyush Jain**

Final-year student | NLP & Transformer-based Applications  
GitHub: https://github.com/piyushjdev
