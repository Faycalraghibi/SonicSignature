import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_and_preprocess_subset(filepath):
    """Loads and preprocesses the subset dataset."""
    df = pd.read_csv(filepath)
    
    # Parse genres from string to list
    df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    
    # Extract year from release_date if year column missing
    if 'year' not in df.columns and 'release_date' in df.columns:
        df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    
    return df

def predict_popularity(df):
    """Predicts song popularity using regression."""
    print("\n--- Popularity Prediction ---")
    
    # Select features
    numeric_features = ['acousticness', 'danceability', 'energy', 'duration_ms', 
                        'instrumentalness', 'valence', 'tempo', 'liveness', 
                        'loudness', 'speechiness', 'year']
    
    # Ensure all features exist
    existing_features = [col for col in numeric_features if col in df.columns]
    
    # Drop rows with missing values in these features
    df_clean = df.dropna(subset=existing_features + ['popularity'])
    
    X = df_clean[existing_features]
    y = df_clean['popularity']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    # using Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Feature Ranking:")
    for f in range(X.shape[1]):
        print(f"{f+1}. {numeric_features[indices[f]]} ({importances[indices[f]]:.4f})")
    
    return model

def analyze_genres(df, output_dir='outputs'):
    """Analyzes and visualizes genre distribution."""
    print("\n--- Genre Analysis ---")
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Flatten genres
    all_genres = [genre for genres in df['genres'] for genre in genres]
    genre_counts = pd.Series(all_genres).value_counts()
    
    print(f"Total unique genres: {len(genre_counts)}")
    print("Top 10 Genres:")
    print(genre_counts.head(10))
    
    # Plot Top 20 Genres
    plt.figure(figsize=(12, 6))
    sns.barplot(x=genre_counts.head(20).values, y=genre_counts.head(20).index, palette='viridis')
    plt.title('Top 20 Genres in Subset')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_genres.png')
    print(f"Saved top_genres.png to {output_dir}")
    plt.close()

def visualize_data(df, output_dir='outputs'):
    """Visualizes data using PCA and t-SNE."""
    print("\n--- Data Visualization ---")
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    numeric_features = ['acousticness', 'danceability', 'energy', 'duration_ms', 
                        'instrumentalness', 'valence', 'tempo', 'liveness', 
                        'loudness', 'speechiness']
    
    X = df[numeric_features].dropna()
    # Normalize
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=10)
    plt.title('PCA of Song Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig(f'{output_dir}/pca_plot.png')
    print(f"Saved pca_plot.png to {output_dir}")
    plt.close()

if __name__ == "__main__":
    file_path = 'dataset/spotify_dataset_subset.csv'
    df = load_and_preprocess_subset(file_path)
    
    predict_popularity(df)
    analyze_genres(df)
    visualize_data(df)
