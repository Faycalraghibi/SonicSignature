import argparse
import sys
import os

# Ensure src module is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description="AARES Project: Sonic Signature")
    parser.add_argument('--part', type=str, choices=['1', '2', 'bonus'], required=True, 
                        help='Select the part of the project to run (1, 2, or bonus)')
    
    args = parser.parse_args()
    
    if args.part == '1':
        print("=== Running Part 1: Classification, Analysis & Recommendation ===")
        
        # Classification
        from src.classification import load_data, preprocess_data, train_model, generate_predictions
        print("\n[Classification]")
        train_path = 'dataset/spotify_dataset_train.csv'
        test_path = 'dataset/spotify_dataset_test.csv'
        if os.path.exists(train_path) and os.path.exists(test_path):
            train_df, test_df = load_data(train_path, test_path)
            X_train, y_train, X_test, _ = preprocess_data(train_df, test_df)
            model = train_model(X_train, y_train)
            generate_predictions(model, X_test, test_df)
        else:
            print("Error: Dataset files not found.")
            
        # Analysis
        from src.analysis import load_and_preprocess_subset, predict_popularity, analyze_genres, visualize_data
        print("\n[Analysis]")
        subset_path = 'dataset/spotify_dataset_subset.csv'
        if os.path.exists(subset_path):
            df_subset = load_and_preprocess_subset(subset_path)
            predict_popularity(df_subset)
            analyze_genres(df_subset)
            visualize_data(df_subset)
        else:
            print("Error: Subset dataset not found.")
            
        # Recommendation
        from src.recommendation import Recommender
        print("\n[Recommendation]")
        rec_path = 'dataset/recommendation_spotify.csv'
        if os.path.exists(rec_path):
            try:
                recommender = Recommender(rec_path)
                # Recommend for the first song in dataset as demo
                sample_id = recommender.df['id'].iloc[0]
                print(f"Generating recommendations for song ID: {sample_id}")
                recs = recommender.get_recommendations(sample_id)
                for i, rec in enumerate(recs):
                    print(f"{i+1}: {rec['id']} (Score: {rec['similarity']:.4f})")
            except Exception as e:
                print(f"Recommendation Error: {e}")
        else:
             print("Error: Recommendation dataset not found.")

    elif args.part == '2':
        print("=== Running Part 2: Audio Fingerprinting ===")
        from src.database import AudioDatabase
        
        db = AudioDatabase()
        
        # Check if database exists, if not create it
        if not os.path.exists(db.db_path):
            print("Database not found. Creating from 'songs/' directory...")
            if os.path.exists('songs'):
                db.create_database('songs')
            else:
                 print("Error: 'songs' directory not found. Cannot create database.")
                 return
        else:
            print("Loading existing database...")
            db.load_database()
            
        # Demo Search
        # We need an excerpt. Let's try to search for one of the files in songs/ if available
        import glob
        songs = glob.glob('songs/*.mp3') + glob.glob('songs/*.wav')
        if songs:
            test_song = songs[0]
            print(f"\nSearching for song: {test_song}")
            # In a real scenario, we would crop this song to 5-15 seconds
            # db.search_excerpt(test_song) 
            # Ideally we'd have a separate 'test_excerpts' folder.
            # For now, let's just run search on the full song as a functionality test
            matches = db.search_excerpt(test_song)
            print(f"Matches found: {matches}")
        else:
            print("No songs found to test search.")

    elif args.part == 'bonus':
        print("=== Running Bonus: Conformal Prediction ===")
        from src.conformal import run_conformal_prediction
        run_conformal_prediction()

if __name__ == "__main__":
    main()
