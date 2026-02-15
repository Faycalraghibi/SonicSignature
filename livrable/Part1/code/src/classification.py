import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import warnings

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    """
    Preprocesses the data:
    - Drops irrelevant columns (id, artist_name, track_name, release_date).
    - Encodes categorical features.
    - Scales numerical features.
    - Handles missing values.
    """
    if 'genre' in train_df.columns:
        y_train = train_df['genre']
        X_train = train_df.drop(columns=['genre'])
    else:
        raise ValueError("Train dataset must contain 'genre' column.")

    drop_cols = ['id', 'artist_name', 'track_name', 'release_date']
    X_train = X_train.drop(columns=drop_cols, errors='ignore')
    X_test = test_df.drop(columns=drop_cols, errors='ignore')
    numeric_features = ['acousticness', 'danceability', 'energy', 'duration_ms', 
                        'instrumentalness', 'valence', 'popularity', 'tempo', 
                        'liveness', 'loudness', 'speechiness', 'year']
    
    categorical_features = ['key', 'mode', 'explicit']

    numeric_features = [col for col in numeric_features if col in X_train.columns]
    categorical_features = [col for col in categorical_features if col in X_train.columns]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, y_train, X_test_processed, preprocessor

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_micro')
    print(f"Cross-Validation F1 Micro Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    clf.fit(X_train, y_train)
    return clf

def generate_predictions(model, X_test, test_df, output_path='outputs/submission.csv'):
    """Generates predictions for the test set and saves them."""
    predictions = model.predict(X_test)
    submission_df = test_df.copy()
    submission_df['genre'] = predictions
    
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return submission_df

if __name__ == "__main__":
    train_path = 'dataset/spotify_dataset_train.csv'
    test_path = 'dataset/spotify_dataset_test.csv'
    
    print("Loading data...")
    train_df, test_df = load_data(train_path, test_path)
    
    print("Preprocessing...")
    X_train, y_train, X_test, _ = preprocess_data(train_df, test_df)
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    print("Generating predictions...")
    generate_predictions(model, X_test, test_df)
