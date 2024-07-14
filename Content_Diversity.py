import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load content data
content_data = pd.read_csv('content_data.csv')

# Calculate content similarity
similarity_matrix = cosine_similarity(content_data)

# Diversify recommendations
def diversify_recommendations(user_id, top_n=10):
    user_history = user_data[user_data['user_id'] == user_id]
    recommendations = []

    for item in user_history['item_id']:
        similar_items = np.argsort(similarity_matrix[item])[-top_n:]
        recommendations.extend(similar_items)

    return np.unique(recommendations)

# Get diversified recommendations for a user
diversified_recs = diversify_recommendations(user_id=123)
print(f'Diversified Recommendations: {diversified_recs}')
