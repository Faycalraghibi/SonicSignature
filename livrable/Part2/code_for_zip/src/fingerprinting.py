import os
import glob
import pickle
import numpy as np
import librosa
from numba import jit

# Import professor-provided functions from utils_projet.py
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils_projet import get_maxima, search_song


def compute_spectrogram(x, sr=3000):
    """
    Compute the energy spectrogram |STFT|^2 of signal *x*.

    Parameters
    ----------
    x  : np.ndarray - mono audio signal
    sr : int - sampling rate (default 3000 Hz)

    Returns
    -------
    S  : np.ndarray - squared-magnitude spectrogram
    """
    n_fft = 2048
    hop_length = 512
    win_length = 1024
    D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    return np.abs(D) ** 2


@jit(nopython=True)
def get_maxima_in_tz(S, maxima, anchor):
    """
    Return up to 10 strongest maxima inside the Target Zone of *anchor*.

    Target Zone definition
    ----------------------
    - Time  : [t_anchor + 10, t_anchor + 300]
    - Freq  : [f_anchor - 10, f_anchor + 10]  (20 px band)
    """
    f_anchor, t_anchor = anchor[0], anchor[1]
    t_min = t_anchor + 10
    t_max = t_anchor + 300
    f_min = max(0, f_anchor - 10)
    f_max = min(S.shape[0], f_anchor + 10)

    # count candidates
    count = 0
    for i in range(len(maxima)):
        f, t = maxima[i][0], maxima[i][1]
        if t_min <= t <= t_max and f_min <= f <= f_max:
            count += 1
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)

    # collect
    candidate_indices = np.empty(count, dtype=np.int64)
    cur = 0
    for i in range(len(maxima)):
        f, t = maxima[i][0], maxima[i][1]
        if t_min <= t <= t_max and f_min <= f <= f_max:
            candidate_indices[cur] = i
            cur += 1

    # amplitudes & top-10
    candidate_amps = np.empty(count, dtype=np.float64)
    candidate_coords = np.empty((count, 2), dtype=np.int64)
    for k in range(count):
        idx = candidate_indices[k]
        f_val = int(maxima[idx][0])
        t_val = int(maxima[idx][1])
        candidate_amps[k] = S[f_val, t_val]
        candidate_coords[k, 0] = f_val
        candidate_coords[k, 1] = t_val

    order = np.argsort(candidate_amps)[::-1]
    n_top = min(10, count)
    result = np.empty((n_top, 2), dtype=np.int64)
    for k in range(n_top):
        result[k] = candidate_coords[order[k]]
    return result


def get_hashes(anchor, maxima_in_tz):
    """
    Build hashes for every (anchor, target) pair.
    Hash = f_anchor * 10^6  +  f_target * 10^3  +  (t_target - t_anchor)

    Returns
    -------
    list of (hash_value, t_anchor) tuples.
    """
    hashes = []
    f_anchor, t_anchor = int(anchor[0]), int(anchor[1])
    for pt in maxima_in_tz:
        f_pt, t_pt = int(pt[0]), int(pt[1])
        h = f_anchor * 1_000_000 + f_pt * 1_000 + (t_pt - t_anchor)
        hashes.append((h, t_anchor))
    return hashes


def process_signal(x, sr=3000):
    """
    Full fingerprinting pipeline: signal -> spectrogram -> maxima -> hashes.

    Returns
    -------
    dict : {hash_value: t_anchor, ...}
    """
    S = compute_spectrogram(x, sr=sr)
    maxima = get_maxima(S)  # from utils_projet
    hashes_dict = {}
    for i in range(len(maxima)):
        anchor = maxima[i]
        tz_maxima = get_maxima_in_tz(S, maxima, anchor)
        for h, t in get_hashes(anchor, tz_maxima):
            hashes_dict[h] = t
    return hashes_dict


class AudioDatabase:
    """Pickle-backed fingerprint store with creation, loading and search."""

    def __init__(self, db_path="dataset/dataset.pickle"):
        self.db_path = db_path
        self.database = []
        self.song_names = []

    def create_database(self, songs_dir="songs"):
        """Walk *songs_dir*, fingerprint every audio file, persist to pickle."""
        print(f"Creating database from {songs_dir}...")
        audio_files = []
        for ext in ("*.mp3", "*.wav", "*.flac"):
            audio_files.extend(glob.glob(os.path.join(songs_dir, ext)))
        print(f"Found {len(audio_files)} audio files.")

        for fp in audio_files:
            try:
                print(f"  Processing {fp}...")
                y, sr = librosa.load(fp, sr=3000)
                hashes = process_signal(y, sr=sr)
                self.database.append(hashes)
                self.song_names.append(os.path.basename(fp))
            except Exception as exc:
                print(f"  Error processing {fp}: {exc}")

        with open(self.db_path, "wb") as fh:
            pickle.dump({"hashes": self.database, "names": self.song_names}, fh)
        print(f"Database saved to {self.db_path} ({len(self.database)} songs).")

    def load_database(self):
        """Load an existing pickle database. Returns True on success."""
        if not os.path.exists(self.db_path):
            print("Database file not found.")
            return False
        with open(self.db_path, "rb") as fh:
            data = pickle.load(fh)
            self.database = data["hashes"]
            self.song_names = data["names"]
        print(f"Loaded database with {len(self.database)} songs.")
        return True

    def search_excerpt(self, excerpt_path):
        """Load an audio excerpt, fingerprint it, and return the top-3 matches."""
        if not self.database:
            if not self.load_database():
                return []
        print(f"Searching for excerpt: {excerpt_path}")
        try:
            y, sr = librosa.load(excerpt_path, sr=3000)
            song_hashes = process_signal(y, sr=sr)
            top_indices = search_song(self.database, song_hashes)
            print("Top 3 Matches:")
            results = []
            for i in top_indices:
                if i < len(self.song_names):
                    print(f"  - {self.song_names[i]}")
                    results.append(self.song_names[i])
            return results
        except Exception as exc:
            print(f"Error searching: {exc}")
            return []
