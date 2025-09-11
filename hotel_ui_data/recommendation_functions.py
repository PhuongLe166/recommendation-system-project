
import json
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_hotel_data():
    """Load hotel data and models"""
    with open('hotels.json', 'r', encoding='utf-8') as f:
        hotels = json.load(f)
    
    with open('hotel_id_mapping.json', 'r', encoding='utf-8') as f:
        id_mapping = json.load(f)
    
    tfidf_sim = np.load('tfidf_similarity.npy')
    
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return hotels, id_mapping, tfidf_sim, vectorizer

def get_recommendations(hotel_id, n_recommendations=5):
    """Get hotel recommendations"""
    hotels, id_mapping, similarity_matrix, _ = load_hotel_data()
    
    if str(hotel_id) not in id_mapping:
        return None
    
    hotel_idx = id_mapping[str(hotel_id)]
    sim_scores = similarity_matrix[hotel_idx]
    similar_indices = np.argsort(sim_scores)[::-1][1:n_recommendations+1]
    
    recommendations = []
    for idx in similar_indices:
        hotel_data = next(h for h in hotels if str(h['Hotel_ID']) == list(id_mapping.keys())[idx])
        recommendations.append({
            'hotel_id': hotel_data['Hotel_ID'],
            'name': hotel_data['Hotel_Name'],
            'similarity_score': float(sim_scores[idx]),
            'total_score': hotel_data['Total_Score_clean']
        })
    
    return recommendations

# Example usage:
# recommendations = get_recommendations(hotel_id=1, n_recommendations=5)
