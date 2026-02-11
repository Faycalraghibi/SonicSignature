# AARES Project: Sonic Signature

## Overview
This project implements a complete pipeline for:
1.  **Spotify Genre Classification**: Predicting musical genres using audio features.
2.  **Popularity Prediction**: Analyzing factors influencing song popularity.
3.  **Song Recommendation**: Generating playlists based on song similarity.
4.  **Audio Fingerprinting**: A Shazam-like system using spectrogram maxima hashing.
5.  **Conformal Prediction**: Quantifying uncertainty in genre predictions (Bonus).

## Structure
- `dataset/`: Contains Spotify datasets and audio files.
- `src/`: Source code modules.
  - `classification.py`: Classification models (ex 1).
  - `analysis.py`: Data analysis and visualization (ex 2).
  - `recommendation.py`: Recommendation engine (ex 3).
  - `fingerprinting.py`: Audio hashing logic (ex 4).
  - `database.py`: Fingerprint database (ex 5).
  - `conformal.py`: Uncertainty quantification (bonus).
- `outputs/`: Generated plots and results.
- `main.py`: Main entry point to run the project.

## Installation
Run the setup script to create a virtual environment and install dependencies:
```powershell
./clean_setup.ps1
```
Or manually:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage
Run the main script with specific flags to execute different parts of the project:

### Part 1: Classification & Analysis
```bash
python main.py --part 1
```

### Part 2: Audio Fingerprinting
```bash
python main.py --part 2
```

### Bonus: Conformal Prediction
```bash
python main.py --part bonus
```
