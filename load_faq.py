import json
from text_process import preprocess_text

filepath = r"D:\VS Work\CodeAlpha\Faq_rag\mobile_faqs.json"

def load_faqs(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        faqs = json.load(f)
    return faqs


def preprocess_all_questions(faqs):
    """
    Har FAQ ke 'question' ko preprocess karta hai aur
    ek naya list return karta hai jisme har entry mein
    original question, processed tokens, answer, aur department ho.
    """
    processed_faqs = []

    for faq in faqs:
        processed_tokens = preprocess_text(faq['question'])

        entry = {
            'question': faq['question'],
            'processed': processed_tokens,
            'answer': faq['answer'],
            'department': faq['department']
        }

        processed_faqs.append(entry)

    return processed_faqs


# Test karne ke liye
if __name__ == "__main__":
    faqs = load_faqs(filepath)
    processed = preprocess_all_questions(faqs)
    print(processed[0])   # sirf pehli entry print karo dekhne ke liye