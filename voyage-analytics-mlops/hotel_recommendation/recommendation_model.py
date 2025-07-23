#hotel_recommendation\recommendation_model.py

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

# Load dataset
df = pd.read_csv("data/hotels.csv")

# Fix date parsing
df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
df = df.dropna(subset=["date"])

# Clean missing values safely
df["name"] = df["name"].fillna("Unknown Hotel")
df["place"] = df["place"].fillna("Unknown Location")

# Optional: drop duplicates (reduce TF-IDF matrix size)
df.drop_duplicates(subset=["name", "place"], inplace=True)

# Optional: sample if very large (for local testing)
if len(df) > 10000:
    df = df.sample(n=5000, random_state=42)

# Combine hotel info text
df["Hotel_Info"] = df["name"] + " | " + df["place"]

# TF-IDF vectorizer with limits (reduce dimensionality)
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["Hotel_Info"])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Index mapping for lookups
indices = pd.Series(df.index, index=df["name"]).drop_duplicates()

# Save everything
os.makedirs("model", exist_ok=True)
joblib.dump(df, "model/hotel_data_df.pkl")
joblib.dump(tfidf_matrix, "model/tfidf_matrix.pkl")
joblib.dump(cosine_sim, "model/cosine_similarity.pkl")
joblib.dump(indices, "model/hotel_indices.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("✅ Model created and saved successfully.")
