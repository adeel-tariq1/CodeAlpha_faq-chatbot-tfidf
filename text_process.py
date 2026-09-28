import nltk
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


lemmatizer = WordNetLemmatizer()

# Stopwords list in english
stop_words = stopwords.words('english')


def preprocess_text(text):

    # Lowercase
    text = text.lower()

    #removing punctuation
    text = ''.join(ch for ch in text if ch not in string.punctuation)

    # split a string of text into individual words and punctuation marks (known as tokens)
    tokens = word_tokenize(text)

    # Stopwords remove karo
    tokens = [token for token in tokens if token.lower() not in stop_words]
    # Lemmatize (reduces a word token to its base dictionary form)
    tokens = [lemmatizer.lemmatize(token,pos="v") for token in tokens]

    return tokens


if __name__ == "__main__":
    sample = "why my phone battery is draining quickly?"
    result = preprocess_text(sample)
    print(result)