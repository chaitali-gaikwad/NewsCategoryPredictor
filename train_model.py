# train_model.py
import pandas as pd
import numpy as np
import regex
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import nltk

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv("english_news_dataset.csv")

# Data preprocessing
threshold = 5
class_counts = df['News Categories'].value_counts()
rare_classes = class_counts[class_counts < threshold].index
df['category_grouped'] = df['News Categories'].apply(lambda x: 'Other' if x in rare_classes else x)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    emoji_pattern = regex.compile(r'\p{Emoji}', flags=regex.UNICODE)
    text = emoji_pattern.sub('', text)  # Remove emojis
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)

df["Content"] = df["Content"].apply(clean_text)

# Prepare data for training
X = df['Content']
y = df['category_grouped']

# Encoding labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Save the model and label encoder
joblib.dump(model, 'best_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Model training complete and saved.")