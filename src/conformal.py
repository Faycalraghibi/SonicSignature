from utils_projet import ConformalPrediction
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.classification import preprocess_data

def run_conformal_prediction():
    print("\n--- Conformal Prediction Analysis ---")
    
    # 1. Load Data
    print("Loading data...")
    train_df = pd.read_csv('dataset/spotify_dataset_train.csv')
    test_df = pd.read_csv('dataset/spotify_dataset_test.csv')
    
    # Extract year if missing
    for df in [train_df, test_df]:
        if 'year' not in df.columns and 'release_date' in df.columns:
             df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    
    # Use first 1500 examples as per prompt instructions
    train_subset = train_df.iloc[:1500]
    
    # Preprocess
    # Note: We need X and y separately. 
    # We should reuse the preprocessing logic from classification.py but handle the subset carefully.
    
    # Let's extract features manually or use the pipeline.
    # To use the pipeline properly, we should fit on the subset.
    
    # For simplicity and robustness, let's just select numeric columns and simple imputation 
    # as KNN doesn't handle missing values natively well without imputation.
    
    feature_cols = ['acousticness', 'danceability', 'energy', 'duration_ms', 
                    'instrumentalness', 'valence', 'popularity', 'tempo', 
                    'liveness', 'loudness', 'speechiness', 'year']
    
    X_train = train_subset[feature_cols].fillna(0).values
    y_train = train_subset['genre'].values
    
    # Select a test point
    X_test = test_df[feature_cols].fillna(0).values
    x_new = X_test[0] # Pick the first test example
    
    # Scale data (KNN is distance based)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    x_new_scaled = scaler.transform(x_new.reshape(1, -1))[0]
    
    # 2. Train KNN
    print("Training KNN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # 3. Create Conformal Predictor
    print("Initializing Conformal Prediction...")
    predictor = ConformalPrediction(knn, X_train_scaled, y_train)
    
    # 4. Predict (Compute p-values)
    print("Computing p-values for a new data point...")
    # This might be slow due to the loop in utils_projet
    predictor.predict(x_new_scaled)
    
    # 5. Analyze Intervals
    print("\nAnalyzing Prediction Intervals:")
    eps_values = [0.05, 0.1, 0.2]
    
    for eps in eps_values:
        interval = predictor.compute_interval(eps)
        print(f"Eps: {eps} (Confidence: {1-eps:.0%}) -> Interval: {interval}")
        
    # 6. Find eps for single genre
    print("\nFinding epsilon for single genre prediction...")
    for eps in np.linspace(0.01, 0.5, 50):
        interval = predictor.compute_interval(eps)
        if len(interval) == 1:
            print(f"Found single genre interval at eps={eps:.2f} (Confidence: {1-eps:.0%}): {interval}")
            break
            
if __name__ == "__main__":
    run_conformal_prediction()
