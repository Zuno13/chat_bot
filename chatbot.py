import json
import pickle
import random
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4")
]

for path, resource in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

model = pickle.load(open("c_m.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)


def preprocess(text):
    words = nltk.word_tokenize(text.lower())

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(words)


def chatbot_response(message):

    processed_message = preprocess(message)

    X = vectorizer.transform([processed_message])

    prediction = model.predict(X)

    tag = encoder.inverse_transform(prediction)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I couldn't understand your question."


if __name__ == "__main__":

    print("=" * 50)
    print("     BVIT STUDENT INFORMATION CHATBOT")
    print("=" * 50)
    print("Type 'quit' to exit.\n")

    while True:

        message = input("You: ")

        if message.lower() in ["quit", "exit"]:
            print("Bot: Thank you for using the chatbot. Goodbye!")
            break

        response = chatbot_response(message)

        print("Bot:", response)