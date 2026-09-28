import streamlit as st
import time

from load_faq import load_faqs, preprocess_all_questions, filepath
from vectorize import build_vectorizer, chatbot_response


@st.cache_resource
def setup():
    faqs = load_faqs(filepath)
    processed = preprocess_all_questions(faqs)
    vectorizer, faq_vectors, corpus = build_vectorizer(processed)
    return processed, vectorizer, faq_vectors


processed, vectorizer, faq_vectors = setup()

def stream_text(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.03)

st.set_page_config(page_title="Mobile FAQ Chatbot", page_icon="📱", layout="centered")

st.title("📱 Mobile Phone FAQ Chatbot")
st.caption("Ask me anything about warranty, returns, shipping, billing, and more.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your FAQ assistant. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "score" in msg:
            st.caption(f"Confidence: {msg['score']:.2f}")

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer, score = chatbot_response(user_input, vectorizer, faq_vectors, processed)

    st.session_state.messages.append({"role": "assistant", "content": answer, "score": score})
    with st.chat_message("assistant"):
        st.write(stream_text(answer))
        st.caption(f"Confidence: {score:.2f}")

with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot answers common mobile phone questions using "
        "TF-IDF and cosine similarity to match your question with the "
        "closest FAQ in the database."
    )
    if st.button("🔄 Clear conversation"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm your FAQ assistant. How can I help you today?"}
        ]
        st.rerun()