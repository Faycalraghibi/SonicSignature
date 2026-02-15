import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("Part 1: Classification, Analysis & Recommendation")
    
    # Classification
    from src.classification import load_data, preprocess_data, train_model, generate_predictions
    print("\n[Classification]")
    train_path = 'dataset/spotify_dataset_train.csv'
    test_path = 'dataset/spotify_dataset_test.csv'
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df, test_df = load_data(train_path, test_path)
        X_train, y_train, X_test, _ = preprocess_data(train_df, test_df)
        model = train_model(X_train, y_train)
        generate_predictions(model, X_test, test_df, output_path='livrable/Part1/submission.csv')
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
            sample_id = recommender.df['id'].iloc[0]
            print(f"Generating recommendations for song ID: {sample_id}")
            recs = recommender.get_recommendations(sample_id)
            for i, rec in enumerate(recs):
                print(f"{i+1}: {rec['id']} (Score: {rec['similarity']:.4f})")
        except Exception as e:
            print(f"Recommendation Error: {e}")
    else:
         print("Error: Recommendation dataset not found.")

if __name__ == "__main__":
    main()
