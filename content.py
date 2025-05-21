from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load data
df = pd.read_csv('movies_ratings_cleaned.csv').drop_duplicates(subset=['title'])

def get_content_based_recommendations(user_id, movie_idx, content_weight=0.6):
    # Assuming df contains movie data with 'genres_cleaned'
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['genres_cleaned'])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get similarity scores for the selected movie
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations (excluding the movie itself)
    top_n = sim_scores[1:11]  # Get the top 10 recommendations (excluding the movie itself)
    
    # Get both title and genres_cleaned for the recommendations
    recommendations = [(df.iloc[i]['title'], df.iloc[i]['genres_cleaned'], score) for i, score in top_n]
    
    return recommendations
