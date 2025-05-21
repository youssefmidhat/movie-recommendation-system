from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv('movies_ratings_cleaned.csv')

def get_svd_recommendations(user_id, movie_idx, collaborative_weight=0.6):
    # Create user-item matrix
    R_df = df.pivot_table(index='userId', columns='movieId', values='rating')
    R_df.fillna(0, inplace=True)  # Fill missing values with 0 (no rating)
    
    # Convert to CSR matrix format for svds
    R = csr_matrix(R_df.values)
    
    # Perform Singular Value Decomposition (SVD)
    U, sigma, Vt = svds(R, k=50)  # k is the number of latent features
    
    # Convert sigma to diagonal matrix
    sigma = np.diag(sigma)
    
    # Predict ratings for all movies for the user
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    
    # Get the predicted rating for the selected movie
    predicted_rating = predicted_ratings[user_id - 1, movie_idx]
    
    # Get top N recommendations based on predicted ratings
    recommendations = []
    
    # Iterate through all movies and get their predicted ratings
    for idx in range(predicted_ratings.shape[1]):
        # Get the movieId from column index (idx)
        movie_id = R_df.columns[idx]  # Get the actual movieId
        predicted_score = predicted_ratings[user_id - 1, idx]  # Predicted rating
        
        # Get the movie title using the movieId
        movie_title = df[df['movieId'] == movie_id]['title'].values[0]
        movie_genres = df[df['movieId'] == movie_id]['genres_cleaned'].values[0]
        
        recommendations.append((movie_title, movie_genres, predicted_score))
    
    # Sort recommendations by predicted rating in descending order
    recommendations.sort(key=lambda x: x[2], reverse=True)
    
    # Return top 10 recommendations
    return recommendations[:10]
