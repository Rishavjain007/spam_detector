# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Create small local SMS dataset ---
data = {
    "label": [
        "ham", "spam", "ham", "spam", "ham",
        "spam", "ham", "spam", "ham", "spam",
        "ham", "ham", "spam", "ham", "spam",
        "ham", "spam", "ham", "ham", "spam"
    ],
    "message": [
        "Hey, are you coming to the party tonight?",
        "Congratulations! You won a free vacation. Reply WIN to claim.",
        "Can we meet tomorrow for lunch?",
        "You’ve been selected for a $500 gift card! Click the link now.",
        "Don’t forget our meeting at 3 PM.",
        "Win big prizes by joining our contest. Click here now!",
        "I’ll call you once I’m free.",
        "Your mobile number has won $10,000! Claim immediately.",
        "Happy Birthday! Hope you have an amazing day!",
        "Get cheap loans instantly. Apply online now.",
        "Let’s catch up this weekend.",
        "Dinner at 8?",
        "Exclusive offer! Get 70% off your next order.",
        "I’ll text you the address.",
        "You’ve won a new iPhone! Claim your reward today.",
        "How was your trip?",
        "You’ve been pre-approved for a credit card! Apply today.",
        "Movie night tonight?",
        "Please call me when you’re done.",
        "Congratulations! You’re our lucky winner!"
    ]
}

df = pd.DataFrame(data)

# --- Prepare data ---
X = df['message']
y = df['label'].map({'ham': 0, 'spam': 1})  # encode labels

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Train Model ---
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Save model & vectorizer ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model and vectorizer saved successfully in 'model/' folder!")