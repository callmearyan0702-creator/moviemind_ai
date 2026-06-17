import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model and vectorizer
model = joblib.load("model.pkl")
cv = joblib.load("countvectorizer.pkl")

ps = PorterStemmer()

# --------------------------
# Your preprocessing functions
# --------------------------

def cleanhtml(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def removeschar(text):
    for char in text:
        if char.isalnum() or char.isspace():
            continue
        text = text.replace(char, '')
    return text

def removestopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join(
        [word for word in text.split()
         if word.lower() not in stop_words]
    )

def stemming(text):
    return " ".join(
        [ps.stem(word) for word in text.split()]
    )

def preprocess(text):
    text = text.lower()
    text = cleanhtml(text)
    text = removeschar(text)
    text = removestopwords(text)
    text = stemming(text)
    return text

# --------------------------
# Streamlit UI
# --------------------------

st.title("🎬 IMDB Sentiment Analysis")

review = st.text_area(
    "Enter Movie Review",
    height=200
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        processed_review = preprocess(review)

        vector = cv.transform([processed_review])

        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")