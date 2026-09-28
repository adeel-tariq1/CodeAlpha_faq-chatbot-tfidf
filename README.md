# 📱 Mobile Phone FAQ Chatbot

A simple FAQ chatbot that answers common mobile phone questions using **NLP-based text matching** (TF-IDF + cosine similarity), with a chat-style interface built in **Streamlit**.

---

## Overview

This project was built as an internship task. The chatbot:

- Collects FAQs related to mobile phones (questions and their answers).
- Preprocesses text using **NLTK** (lowercasing, punctuation removal, tokenization, stopword removal, lemmatization).
- Matches a user's question with the most similar FAQ using **TF-IDF vectorization** and **cosine similarity**.
- Displays the best matching answer, along with a confidence score.
- Falls back to a polite "not relevant" message when no FAQ is a close enough match.
- Provides a simple chat UI built with Streamlit.

---

## FAQ Categories

The FAQ dataset covers 7 departments:

1. Technical Support
2. Warranty & Repair
3. Billing/Payment
4. Shipping/Delivery
5. Returns & Refunds
6. Sales/Pre-Purchase
7. General/Account

---

## Project Structure

```
Faq_rag/
│
├── mobile_faqs.json      # FAQ dataset (question, answer, department)
├── text_process.py       # Text preprocessing (tokenize, clean, lemmatize)
├── load_faq.py           # Loads FAQ data and applies preprocessing
├── main.py                # TF-IDF vectorizer, matching logic, terminal chatbot
├── faq_app.py             # Streamlit chat frontend
└── README.md
```

---

## How It Works

1. **Preprocessing** (`text_process.py`): Each question is lowercased, stripped of punctuation, tokenized, cleaned of stopwords, and lemmatized.
2. **Vectorization** (`main.py`): All FAQ questions are converted into TF-IDF vectors, capturing which words matter most.
3. **Matching**: A user's question goes through the same preprocessing and vectorization, then is compared against every FAQ vector using cosine similarity. The closest match is returned.
4. **Confidence threshold**: If the best similarity score is below `0.3`, the chatbot responds that the question seems irrelevant instead of guessing.

---

## Setup & Installation

1. Install the required libraries:

```bash
pip install nltk scikit-learn pandas streamlit
```

2. Download the required NLTK data (run once in Python):

```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## Running the App

**Option 1: Terminal chatbot**

```bash
python main.py
```

Type your question, or type `exit` to quit.

**Option 2: Streamlit chat interface (recommended)**

```bash
streamlit run faq_app.py
```

This opens a browser-based chat window where you can ask questions and see the bot's answer along with its confidence score.

---

## Example

**User:** How long is the warranty of my phone?
**Bot:** All new phones come with a standard 12-month warranty from the date of purchase, covering manufacturing defects.
**Confidence:** 0.69

---

## Notes / Possible Improvements

- Add more FAQs per department to improve matching accuracy.
- Verb-aware lemmatization (`pos='v'`) for better word normalization.
- Expand to a larger, real-world FAQ dataset.
