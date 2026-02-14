"""
Audio Fingerprinting System (Shazam-like)
=========================================
A complete audio fingerprinting pipeline for song identification.

Implements:
  1. Spectrogram computation via STFT (n_fft=2048, hop=512, win=1024, sr=3000)
  2. Local-maxima extraction with amplitude filtering and distance pruning
  3. Target-zone hashing (300 px time x 20 px freq, top-10 per anchor)
  4. Pickle-based fingerprint database with histogram-based search

Author : Faycal Raghibi
Course : AARES – IMT Nord Europe
Date   : February 2026
"""

import os
import glob
import pickle
import numpy as np
import librosa
from numba import jit


# --------------------------------------------------------------------------- #
#  Spectrogram maxima utilities (from utils_projet.py – J. Miramont)          #
# --------------------------------------------------------------------------- #

@jit(nopython=True)
def filter_by_distance(order, maxima, idx, idx2):
    """Remove maxima that are closer than 10 pixels to a stronger peak."""
    for k, i in enumerate(order):
        for q in range(k, len(order)):
            j = order[q]
            d = (
                (maxima[idx2[i], 0] - maxima[idx2[j], 0]) ** 2
                + (maxima[idx2[i], 1] - maxima[idx2[j], 1]) ** 2
            ) ** 0.5
            if d > 0 and d < 10:
                if idx[idx2[i]]:
                    idx[idx2[j]] = False
    return idx


def filter_maxima(maxima, S):
    """Keep only peaks above the 90th-percentile amplitude, then prune by distance."""
    idx = np.zeros((len(maxima),), dtype=bool)
    thr = np.quantile(S, 0.9)
    for k in range(len(maxima)):
        if S[maxima[k, 0], maxima[k, 1]] > thr:
            idx[k] = True

    idx2 = np.where(idx)[0]
    sel_maxima = maxima[idx]
    amps = np.zeros((len(sel_maxima),))
    for i, _k in enumerate(sel_maxima):
        amps[i] = S[sel_maxima[i, 0], sel_maxima[i, 1]]

    order = np.argsort(amps)[::-1]
    idx = filter_by_distance(order, maxima, idx, idx2)
    return maxima[idx, :]


def get_maxima(S):
    """Find local maxima of spectrogram *S* as 3x3-grid winners."""
    aux_S = np.zeros((S.shape[0] + 2, S.shape[1] + 2)) - np.inf
    aux_S[1:-1, 1:-1] = S
    S = aux_S
    aux_ceros = (
        (S >= np.roll(S, 1, 0))
        & (S >= np.roll(S, -1, 0))
        & (S >= np.roll(S, 1, 1))
        & (S >= np.roll(S, -1, 1))
        & (S >= np.roll(S, [-1, -1], [0, 1]))
        & (S >= np.roll(S, [1, 1], [0, 1]))
        & (S >= np.roll(S, [-1, 1], [0, 1]))
        & (S >= np.roll(S, [1, -1], [0, 1]))
    )
    [y, x] = np.where(aux_ceros == True)  # noqa: E712
    pos = np.zeros((len(x), 2))
    pos[:, 0] = y - 1
    pos[:, 1] = x - 1
    pos = filter_maxima(pos.astype(int), S)
    return pos


def search_song(db_hashes, song_hashes):
    """
    Search for a song in the database by comparing hash dictionaries.

    Returns the indices of the top-3 matches ranked by the largest
    histogram bin of time-offset differences.
    """
    scores = []
    for hashes in db_hashes:
        tokens_present = [t for t in song_hashes if t in hashes]
        offsets = [hashes[t] - song_hashes[t] for t in tokens_present]
        count, _bins = np.histogram(offsets)
        scores.append(np.max(count) if len(count) > 0 else 0)
    return np.argsort(scores)[::-1][:3]


# --------------------------------------------------------------------------- #
#  Fingerprinting core                                                        #
# --------------------------------------------------------------------------- #

def compute_spectrogram(x, sr=3000):
    """
    Compute the energy spectrogram |STFT|^2 of signal *x*.

    Parameters
    ----------
    x  : np.ndarray – mono audio signal
    sr : int – sampling rate (default 3000 Hz)

    Returns
    -------
    S  : np.ndarray – squared-magnitude spectrogram
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

    # --- pass 1: count candidates ---
    count = 0
    for i in range(len(maxima)):
        f, t = maxima[i][0], maxima[i][1]
        if t_min <= t <= t_max and f_min <= f <= f_max:
            count += 1
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)

    # --- pass 2: collect ---
    candidate_indices = np.empty(count, dtype=np.int64)
    cur = 0
    for i in range(len(maxima)):
        f, t = maxima[i][0], maxima[i][1]
        if t_min <= t <= t_max and f_min <= f <= f_max:
            candidate_indices[cur] = i
            cur += 1

    # --- pass 3: amplitudes & top-10 ---
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
    Full fingerprinting pipeline: signal → spectrogram → maxima → hashes.

    Returns
    -------
    dict : {hash_value: t_anchor, ...}
    """
    S = compute_spectrogram(x, sr=sr)
    maxima = get_maxima(S)
    hashes_dict = {}
    for i in range(len(maxima)):
        anchor = maxima[i]
        tz_maxima = get_maxima_in_tz(S, maxima, anchor)
        for h, t in get_hashes(anchor, tz_maxima):
            hashes_dict[h] = t
    return hashes_dict


# --------------------------------------------------------------------------- #
#  Database management                                                        #
# --------------------------------------------------------------------------- #

class AudioDatabase:
    """Pickle-backed fingerprint store with creation, loading and search."""

    def __init__(self, db_path="dataset/dataset.pickle"):
        self.db_path = db_path
        self.database = []   # list of hash dicts
        self.song_names = []

    # ---- build ------------------------------------------------------------ #
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

    # ---- load ------------------------------------------------------------- #
    def load_database(self):
        """Load an existing pickle database.  Returns True on success."""
        if not os.path.exists(self.db_path):
            print("Database file not found.")
            return False
        with open(self.db_path, "rb") as fh:
            data = pickle.load(fh)
            self.database = data["hashes"]
            self.song_names = data["names"]
        print(f"Loaded database with {len(self.database)} songs.")
        return True

    # ---- search ----------------------------------------------------------- #
    def search_excerpt(self, excerpt_path):
        """
        Load an audio excerpt, fingerprint it, and return the top-3 matches.
        """
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


# --------------------------------------------------------------------------- #
#  CLI entry-point                                                            #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Audio fingerprinting: build a database and search excerpts."
    )
    sub = parser.add_subparsers(dest="command")

    # -- build --
    build_p = sub.add_parser("build", help="Build fingerprint database from a folder of songs.")
    build_p.add_argument("--songs-dir", default="songs", help="Directory containing audio files.")
    build_p.add_argument("--db-path", default="dataset/dataset.pickle", help="Output pickle path.")

    # -- search --
    search_p = sub.add_parser("search", help="Search an audio excerpt against the database.")
    search_p.add_argument("excerpt", help="Path to the audio excerpt to identify.")
    search_p.add_argument("--db-path", default="dataset/dataset.pickle", help="Pickle database path.")

    args = parser.parse_args()

    if args.command == "build":
        db = AudioDatabase(db_path=args.db_path)
        db.create_database(songs_dir=args.songs_dir)

    elif args.command == "search":
        db = AudioDatabase(db_path=args.db_path)
        db.search_excerpt(args.excerpt)

    else:
        parser.print_help()
