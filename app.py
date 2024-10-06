# app.py
import streamlit as st
import pandas as pd
import joblib
import re
import string
from nltk.corpus import stopwords
import regex

# Load the trained model and label encoder
model = joblib.load('best_model.pkl')
le = joblib.load('label_encoder.pkl')

def clean_text(text):
    """Clean the input text for prediction."""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    emoji_pattern = regex.compile(r'\p{Emoji}', flags=regex.UNICODE)
    text = emoji_pattern.sub('', text)  # Remove emojis
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)

def predict_news_category(text):
    """Predicts the news category for a given text using the best model."""
    cleaned_text = clean_text(text)
    predicted_label_encoded = model.predict([cleaned_text])[0]
    predicted_label = le.inverse_transform([predicted_label_encoded])[0]
    return predicted_label

# Streamlit UI
st.title("News Category Prediction")

# Text input for the user
input_text = st.text_area("Enter news text to predict its category.", height=200)

# Button to trigger prediction
if st.button("Predict"):
    if input_text:
        predicted_category = predict_news_category(input_text)
        st.success(f"The predicted category is: **{predicted_category}**")
    else:
        st.error("Please enter some text.")

# Optionally add a footer
st.markdown("---")
# st.write("This app predicts the category of news articles based on the content.")
