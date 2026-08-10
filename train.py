import json
import pickle
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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

with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

sentences = []
labels = []

def preprocess(text):
    words = nltk.word_tokenize(text.lower())

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.isalpha() and word not in stop_words
    ]

    return " ".join(words)

for intent in intents["intents"]:
    tag = intent["tag"]

    for pattern in intent["patterns"]:
        sentences.append(preprocess(pattern))
        labels.append(tag)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sentences)

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

model = LogisticRegression(max_iter=2000)

model.fit(X, y)

pickle.dump(model, open("c_m.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

print("=" * 40)
print("Training Completed Successfully")
print("Number of Intents :", len(intents["intents"]))
print("Training Samples  :", len(sentences))
print("=" * 40)