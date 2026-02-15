import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.conformal import run_conformal_prediction

def main():
    print("Bonus: Conformal Prediction with KNN")
    run_conformal_prediction(train_path='dataset/spotify_dataset_train.csv', n_samples=1500)

if __name__ == "__main__":
    main()
