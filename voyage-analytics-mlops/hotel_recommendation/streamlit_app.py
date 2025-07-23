# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

# Load model artifacts
df = joblib.load("model/hotel_data_df.pkl")
cosine_sim = joblib.load("model/cosine_similarity.pkl")
indices = joblib.load("model/hotel_indices.pkl")

# ------------------- UI Setup -------------------
st.set_page_config(
    page_title="Hotel Recommender",
    page_icon="🏨",
    layout="centered"
)

st.title("🏨 Hotel Recommendation System")
st.markdown("Find hotels similar to the one you like based on location and content!")

# Sidebar filters
st.sidebar.header("🔍 Filter Options")

# Hotel name selector
hotel_names = df["name"].unique()
selected_hotel = st.sidebar.selectbox("Select a hotel", sorted(hotel_names))

# Optional: filter by location
places = df["place"].unique()
selected_place = st.sidebar.selectbox("Filter by location", ["All"] + sorted(places.tolist()))

# Optional: filter by max price
max_price = st.sidebar.slider(
    "Max price per day", 
    min_value=float(df["price"].min()), 
    max_value=float(df["price"].max()), 
    value=float(df["price"].max())
)

# Number of recommendations
top_n = st.sidebar.slider("Number of recommendations", 1, 20, 5)

# ------------------- Recommendation Logic -------------------
def get_recommendations(hotel_name, cosine_sim, df, indices, top_n):
    try:
        idx = indices.get(hotel_name)
        if idx is None:
            return pd.DataFrame()

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 10]
        hotel_indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[hotel_indices].copy()

        # Apply optional filters
        if selected_place != "All":
            recommendations = recommendations[recommendations["place"] == selected_place]

        recommendations = recommendations[recommendations["price"] <= max_price]

        return recommendations.head(top_n)[["name", "place", "days", "price", "total"]].reset_index(drop=True)

    except Exception as e:
        # Optional: log the error if needed
        # st.error(f"An error occurred: {e}")
        return pd.DataFrame()

# ------------------- Show Results -------------------
if st.button("🎯 Recommend Hotels"):
    with st.spinner("Finding similar hotels..."):
        results = get_recommendations(selected_hotel, cosine_sim, df, indices, top_n)

    if not results.empty:
        st.success(f"Top {len(results)} recommendations similar to **{selected_hotel}**")
        st.dataframe(results.style.format({"price": "₹{:.2f}", "total": "₹{:.2f}"}))
    else:
        st.warning("😕 No matching hotels found or an error occurred.")

# Footer
st.markdown("---")
st.caption("📘 Created by Harish | M.Sc Data Science | Powered by Streamlit + Scikit-Learn")
