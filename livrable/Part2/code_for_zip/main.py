import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fingerprinting import AudioDatabase

def main():
    print("=== Part 2: Audio Fingerprinting ===")

    db = AudioDatabase(db_path='dataset/dataset.pickle')

    # Build the database from songs/ directory
    if not os.path.exists(db.db_path):
        print("\nDatabase not found. Creating from 'songs/' directory...")
        if os.path.exists('songs'):
            db.create_database('songs')
        else:
            print("Error: 'songs' directory not found. Cannot create database.")
            return
    else:
        print("\nLoading existing database...")
        db.load_database()

    # Search for an excerpt
    songs = glob.glob('songs/*.mp3') + glob.glob('songs/*.wav') + glob.glob('songs/*.flac')
    if songs:
        test_song = songs[0]
        print(f"\nSearching for song: {test_song}")
        matches = db.search_excerpt(test_song)
        print(f"Matches found: {matches}")
    else:
        print("\nNo songs found in 'songs/' to test search.")

if __name__ == "__main__":
    main()
