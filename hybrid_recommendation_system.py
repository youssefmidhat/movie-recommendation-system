from content import get_content_based_recommendations
from collaborative import get_svd_recommendations
import pandas as pd

def hybrid_recommendations(user_id, movie_idx, content_weight=0.6, collaborative_weight=0.6):
    # Get content-based recommendations
    content_recs = get_content_based_recommendations(user_id, movie_idx, content_weight)
    
    # Get collaborative-based recommendations using SVD
    collaborative_recs = get_svd_recommendations(user_id, movie_idx, collaborative_weight)
    
    # Combine the recommendations (weighted averaging approach)
    combined_recs = {}
    
    # Add content-based recommendations to the combined list
    for movie, genre, content_rating in content_recs:
        if movie not in combined_recs:
            combined_recs[movie] = content_rating * content_weight
    
    # Add collaborative-based recommendations to the combined list
    for movie, genre, collaborative_rating in collaborative_recs:
        if movie in combined_recs:
            # If movie already exists in combined_recs, we average the ratings based on the weights
            combined_recs[movie] += collaborative_rating * collaborative_weight
        else:
            combined_recs[movie] = collaborative_rating * collaborative_weight
    
    # Sort the recommendations by combined score
    sorted_recs = sorted(combined_recs.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return [(movie, score) for movie, score in sorted_recs[:10]]
