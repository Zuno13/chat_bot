BVIT STUDENT CHATBOT
====================

Project Title:
Intent Detection Chatbot using Machine Learning

Project Description:
This project is an intent detection chatbot developed using Python and basic
Natural Language Processing techniques. The chatbot is designed to answer
common questions related to Bharati Vidyapeeth Institute of Technology (BVIT),
Kharghar and MSBTE.

Technologies Used:
- Python
- NLTK
- Scikit-learn
- Streamlit
- JSON

Machine Learning:
The chatbot uses CountVectorizer to convert text into numerical features and
Logistic Regression to classify the user's message into a predefined intent.

Files:
app.py - Streamlit user interface
chatbot.py - Chatbot prediction and response system
train.py - Training the machine learning model
intents.json - Intent patterns and responses
requirements.txt - Required Python packages

Model Files:
c_m.pkl
vectorizer.pkl
label_encoder.pkl

How to Run:

1. Create a virtual environment.

2. Activate the environment.

3. Install the required packages:

pip install -r requirements.txt

4. Train the chatbot:

python train.py

5. Start the Streamlit application:

python -m streamlit run app.py

Project Scope:
This is a college-level Data Science project created to demonstrate
Natural Language Processing, text preprocessing, feature extraction and
intent classification using Machine Learning.
