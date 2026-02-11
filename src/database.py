import os
import pickle
import librosa
import numpy as np
import glob
from src.fingerprinting import process_signal
from utils_projet import search_song

class AudioDatabase:
    def __init__(self, db_path='dataset/dataset.pickle'):
        self.db_path = db_path
        self.database = [] # List of dictionaries
        self.song_names = [] # List of song names corresponding to db index
    
    def create_database(self, songs_dir='songs'):
        """
        Creates the fingerprint database from audio files in songs_dir.
        """
        print(f"Creating database from {songs_dir}...")
        
        # Find all audio files
        audio_extensions = ['*.mp3', '*.wav', '*.flac']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(songs_dir, ext)))
            
        print(f"Found {len(audio_files)} audio files.")
        
        for file_path in audio_files:
            try:
                print(f"Processing {file_path}...")
                # Load song (sr=3000 as per prompt)
                y, sr = librosa.load(file_path, sr=3000)
                
                # Process signal to get hashes
                hashes = process_signal(y, sr=sr)
                
                # Store hashes
                self.database.append(hashes)
                self.song_names.append(os.path.basename(file_path))
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        # Save to pickle
        with open(self.db_path, 'wb') as handle:
            pickle.dump({'hashes': self.database, 'names': self.song_names}, handle)
            
        print(f"Database saved to {self.db_path}")
        
    def load_database(self):
        """Loads the database from pickle file."""
        if not os.path.exists(self.db_path):
            print("Database file not found.")
            return False
            
        with open(self.db_path, 'rb') as handle:
            data = pickle.load(handle)
            self.database = data['hashes']
            self.song_names = data['names']
            
        print(f"Loaded database with {len(self.database)} songs.")
        return True

    def search_excerpt(self, excerpt_path):
        """
        Searches for a song excerpt in the database.
        Returns top 3 matches.
        """
        if not self.database:
            if not self.load_database():
                return
        
        print(f"Searching for {excerpt_path}...")
        
        # Load excerpt
        try:
            y, sr = librosa.load(excerpt_path, sr=3000)
            
            # Get hashes
            song_hashes = process_signal(y, sr=sr)
            
            # Search using utils_projet function
            # Note: utils_projet.search_song takes (db_hashes, song_hashes)
            # db_hashes is a list of dictionaries
            
            top_indices = search_song(self.database, song_hashes)
            
            print("Top 3 Matches:")
            for i in top_indices:
                if i < len(self.song_names):
                    print(f"- {self.song_names[i]}")
                else:
                    print(f"- Index {i} (Name not found)")
                    
            return [self.song_names[i] for i in top_indices if i < len(self.song_names)]
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []

if __name__ == '__main__':
    # Test stub
    db = AudioDatabase()
    # db.create_database()
