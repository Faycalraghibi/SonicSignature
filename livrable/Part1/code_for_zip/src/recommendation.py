import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                         'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
        self._prepare_data()

    def _prepare_data(self):
        """Preprocesses data for similarity calculation."""
        # Handle missing values
        self.df = self.df.dropna(subset=self.features)
        
        # Scale features
        self.scaler = StandardScaler()
        self.feature_matrix = self.scaler.fit_transform(self.df[self.features])
        
        # Create ID mapping
        self.id_to_idx = {id_: idx for idx, id_ in enumerate(self.df['id'])}
        self.idx_to_id = {idx: id_ for idx, id_ in enumerate(self.df['id'])}

    def get_recommendations(self, song_id, n=10):
        """
        Returns top n recommended songs based on cosine similarity.
        """
        if song_id not in self.id_to_idx:
            raise ValueError(f"Song ID {song_id} not found in database.")
        
        idx = self.id_to_idx[song_id]
        
        # Calculate similarity for this song against all others
        # Reshape to (1, n_features) for scikit-learn
        target_vector = self.feature_matrix[idx].reshape(1, -1)
        sim_scores = cosine_similarity(target_vector, self.feature_matrix).flatten()
        
        # Get top n indices (excluding self)
        # argsort returns indices of sorted array, we want descending order
        top_indices = np.argsort(sim_scores)[::-1][1:n+1]
        
        recommendations = []
        for i in top_indices:
            rec_id = self.idx_to_id[i]
            rec_song = self.df.iloc[i]
            recommendations.append({
                'id': rec_id,
                'name': rec_song.get('name', 'Unknown'), # Assuming 'name' column exists or similar
                'artist': rec_song.get('artists', 'Unknown'), # Assuming 'artists' column exists
                'similarity': sim_scores[i]
            })
            
        return recommendations

if __name__ == "__main__":
    filepath = 'dataset/recommendation_spotify.csv'
    recommender = Recommender(filepath)
    
    # Test with a random song ID from the dataset
    random_id = recommender.df['id'].iloc[0]
    print(f"Recommendations for song ID: {random_id}")
    recs = recommender.get_recommendations(random_id)
    for i, rec in enumerate(recs):
        print(f"{i+1}: {rec['id']} (Score: {rec['similarity']:.4f})")
