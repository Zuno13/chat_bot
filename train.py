import json
import pickle
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(sentences)

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=2000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

pickle.dump(model, open("c_m.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

print("=" * 40)
print("Training Completed Successfully")
print("Number of Intents :", len(intents["intents"]))
print("Training Samples  :", len(sentences))
print("Testing Samples   :", X_test.shape[0])
print("Model Accuracy    :", round(accuracy * 100, 2), "%")
print("=" * 40)