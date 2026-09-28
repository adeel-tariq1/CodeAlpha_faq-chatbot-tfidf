from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_process import preprocess_text
from load_faq import load_faqs,preprocess_all_questions,filepath


def build_vectorizer(processed_faqs):
    """
    Saare FAQ questions ke processed tokens ko joinkar
    ek string list banata hai, phir TF-IDF vectorizer fit karta hai.
    Returns: vectorizer object, aur FAQ vectors (matrix)
    """
    corpus = [" ".join(entry['processed']) for entry in processed_faqs]

    vectorizer = TfidfVectorizer()
    faq_vectors = vectorizer.fit_transform(corpus)

    return vectorizer, faq_vectors, corpus


def get_best_match(user_question, vectorizer, faq_vectors, processed_faqs):
    """
    User ke sawal ko preprocess + vectorize karta hai,
    phir saari FAQs se compare karke sabse zyada similar wali dhoondta hai.
    """
    user_tokens = preprocess_text(user_question)
    user_string = " ".join(user_tokens)

    user_vector = vectorizer.transform([user_string])

    similarities = cosine_similarity(user_vector, faq_vectors)

    best_index = similarities.argmax()
    best_score = similarities[0][best_index]

    return processed_faqs[best_index], best_score

THRESHOLD = 0.3

def chatbot_response(user_question, vectorizer, faq_vectors, processed_faqs):

    match, score = get_best_match(user_question, vectorizer, faq_vectors, processed_faqs)

    if score < THRESHOLD:
        return "Sorry, It seems like your question is not relevant.Kindly ask any query regarging mobile phone.",score

    return match['answer'],score


def run_chat():
    """
    Interactive loop jo user se sawal leta hai jab tak
    woh 'exit' na type kare.
    """
    faqs = load_faqs(filepath)
    processed = preprocess_all_questions(faqs)
    vectorizer, faq_vectors, corpus = build_vectorizer(processed)

    print("Hi! I’m an FAQ chatbot. How can I help you? (Type 'exit' to close.)")
    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Good Bye! Have a nice day.")
            break

        answer,score = chatbot_response(user_input, vectorizer, faq_vectors, processed)
        print(f"Bot: {answer}")
        print(f"Similarity score: {score:.2f}\n")


if __name__ == "__main__":
    run_chat()
