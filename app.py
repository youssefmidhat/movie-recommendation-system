import streamlit as st
import pandas as pd
from hybrid_recommendation_system import hybrid_recommendations
from content import get_content_based_recommendations
from collaborative import get_svd_recommendations

# Load data
df = pd.read_csv('movies_ratings_cleaned.csv')

# Page configuration
st.set_page_config(page_title="Movie Recommendation System", layout="wide", page_icon=":movie_camera:")

# App title
st.title("🎬 Movie Recommendation System")
st.markdown("""
Choose a recommendation model from the sidebar, and get personalized movie suggestions.
""")

# Sidebar: Step 1 - Select model
st.sidebar.header("1️⃣ Select Recommendation Model")
model_choice = st.sidebar.selectbox("Choose model:", ["-- Select --", "Content-Based", "Collaborative Filtering", "Hybrid"])

# Sidebar: Step 2 - Dynamic inputs based on model
st.sidebar.header("2️⃣ Provide Input")

# Initialize variables
user_id = None
movie_title = None

# Show inputs based on selected model
if model_choice == "Content-Based":
    movie_options = df['title'].unique()
    movie_title = st.sidebar.selectbox("Select a movie:", movie_options)

elif model_choice == "Collaborative Filtering":
    user_id = st.sidebar.number_input("Enter User ID:", min_value=1, max_value=df['userId'].max(), step=1)

elif model_choice == "Hybrid":
    user_id = st.sidebar.number_input("Enter User ID:", min_value=1, max_value=df['userId'].max(), step=1)
    movie_options = df['title'].unique()
    movie_title = st.sidebar.selectbox("Select a movie:", movie_options)

# Sidebar: Number of recommendations
num_recommendations = st.sidebar.slider("Number of recommendations:", 5, 20, 10)

# Button to generate recommendations
if st.sidebar.button("🎯 Get Recommendations"):
    st.subheader(f"{model_choice} Recommendations")
    try:
        recs = []

        if model_choice == "Content-Based" and movie_title:
            movie_idx = df[df['title'] == movie_title].index[0]
            content_recs = get_content_based_recommendations(user_id=1, movie_idx=movie_idx)  # Dummy user_id
            recs = [(title,) for title, _, _ in content_recs]

        elif model_choice == "Collaborative Filtering" and user_id:
            collab_recs = get_svd_recommendations(user_id=user_id, movie_idx=0)  # Dummy movie_idx
            recs = [(title,) for title, _, _ in collab_recs]

        elif model_choice == "Hybrid" and user_id and movie_title:
            movie_idx = df[df['title'] == movie_title].index[0]
            hybrid_recs = hybrid_recommendations(user_id, movie_idx)
            recs = [(title,) for title, _ in hybrid_recs]

        else:
            st.warning("Please make sure you’ve provided the required input(s).")

        # Process and display the results
        recs = recs[:num_recommendations]
        recs_df = pd.DataFrame(recs, columns=["Movie"])
        recs_df["Genres"] = recs_df["Movie"].apply(
            lambda x: df[df['title'] == x]['genres_cleaned'].values[0] if len(df[df['title'] == x]) > 0 else 'Unknown'
        )

        st.dataframe(recs_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
