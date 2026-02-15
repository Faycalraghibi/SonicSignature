import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils_projet import ConformalPrediction


def load_and_prepare_data(train_path, n_samples=1500):
    """Load Spotify training data and prepare features/labels for KNN."""
    df = pd.read_csv(train_path)
    df = df.head(n_samples)

    numeric_features = ['acousticness', 'danceability', 'energy', 'duration_ms',
                        'instrumentalness', 'valence', 'popularity', 'tempo',
                        'liveness', 'loudness', 'speechiness']

    existing_features = [col for col in numeric_features if col in df.columns]

    X = df[existing_features].values
    y = df['genre'].values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le


def run_conformal_prediction(train_path='dataset/spotify_dataset_train.csv', n_samples=1500):
    """
    Run conformal prediction with KNN on the Spotify dataset.
    
    Steps:
    1. Load and preprocess first n_samples from training data
    2. Train a KNN classifier (k=5)
    3. Create ConformalPrediction object from utils_projet
    4. Compute prediction intervals for various epsilon values
    5. Analyze interval sizes and find single-genre threshold
    """
    print("Loading and preparing data...")
    X, y, label_encoder = load_and_prepare_data(train_path, n_samples)
    genres = label_encoder.classes_

    print(f"Dataset: {len(X)} samples, {len(genres)} genres")
    print(f"Genres: {list(genres)}")

    print("\nTraining KNN classifier (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    cp = ConformalPrediction(knn, X, y)

    test_idx = len(X) - 1
    x_test = X[test_idx].reshape(1, -1)
    true_label = genres[y[test_idx]]
    print(f"\nTest sample index: {test_idx}, true genre: {true_label}")

    print("\nComputing p-values (this may take a moment)...")
    cp.predict(x_test)
    pz = cp.pz

    print("\nP-values per genre:")
    for i, genre in enumerate(genres):
        print(f"  {genre}: {pz[i]:.4f}")

    print("\nPrediction Intervals...")
    print(f"{'Epsilon':<12} {'Interval Size':<16} {'Genres in Interval'}")
    print("-" * 60)

    epsilons = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]
    for eps in epsilons:
        interval = cp.compute_interval(eps)
        interval_genres = [genres[i] for i in interval] if len(interval) > 0 else []
        print(f"{eps:<12.2f} {len(interval):<16} {interval_genres}")

    print("\nFinding Single-Genre Threshold...")
    for eps in np.arange(0.01, 1.0, 0.01):
        interval = cp.compute_interval(eps)
        if len(interval) == 1:
            single_genre = genres[interval[0]]
            print(f"Single-genre prediction at epsilon={eps:.2f}: {single_genre}")
            print(f"True genre: {true_label}")
            print(f"Correct: {single_genre == true_label}")
            break
    else:
        print("No single-genre threshold found in [0.01, 1.0)")

    return cp, genres
