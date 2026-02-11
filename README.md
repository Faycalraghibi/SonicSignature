# Sonic Signature: Advanced Music Analysis & Audio Fingerprinting
## AARES Project

![Project Status](https://img.shields.io/badge/status-complete-green.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

This repository contains the complete implementation for the "Sonic Signature" project, bridging Machine Learning with Audio Signal Processing.

## 🎵 Project Overview

The project is divided into two major components carried out in independent Jupyter Notebooks:

### **Part 1: Spotify Machine Learning Analysis**
Focuses on metadata and high-level audio features extracted from the Spotify API.
- **Genre Classification**: Predicting musical genres using Random Forest ($F1 \approx 0.44$).
- **Popularity Prediction**: Regressing song popularity based on audio characteristics ($R^2 \approx 0.21$).
- **Recommendation Engine**: Content-based filtering using Cosine Similarity to generate playlists.

### **Part 2: Audio Fingerprinting & Bonus**
Focuses on raw audio signal processing.
- **Audio Fingerprinting**: A "Shazam-like" system using STFT spectrograms, peak extraction (constellation maps), and hashing for robust song identification.
- **Conformal Prediction**: Uncertainty quantification for the genre classifier, providing prediction sets with guaranteed coverage (e.g., 95% confidence).

---

## 📂 Repository Structure

```
SonicSignature/
├── dataset/                    # CSV datasets (train, test, subset)
├── songs/                      # MP3/WAV files for fingerprinting
├── notebooks/                  # Interactive Jupyter Notebooks
│   ├── Part1_Spotify_ML.ipynb  # Classification, Regression, Recommendation
│   └── Part2_Audio_Fingerprinting_and_Bonus.ipynb # Fingerprinting, Conformal Pred.
├── src/                        # Python Source Modules
│   ├── classification.py       # ML Pipeline logic
│   ├── analysis.py             # Feature analysis & Visualization
│   ├── recommendation.py       # Recommender system implementation
│   ├── fingerprinting.py       # Spectrogram & Hashing logic
│   ├── database.py             # Fingerprint storage & Search
│   └── conformal.py            # Conformal Prediction logic
├── outputs/                    # Generated plots, models, and predictions
├── repport/                    # LaTeX Report files
│   ├── Main.tex                # Main report source
│   └── Main.pdf                # Compiled PDF report
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SonicSignature.git
    cd SonicSignature
    ```

2.  **Set up the environment:**
    We provide a setup script for convenience (Windows PowerShell).
    ```powershell
    ./clean_setup.ps1
    ```
    *Alternatively, manually:*
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    
    pip install -r requirements.txt
    ```

---

## 💻 Usage

The project is designed to be explored via **Jupyter Notebooks**, which contain the code, visualizations, and narrative.

1.  **Launch Jupyter Lab/Notebook:**
    ```bash
    jupyter lab
    ```

2.  **Open the notebooks:**
    - Navigate to the `notebooks/` directory.
    - Open `Part1_Spotify_ML.ipynb` for ML tasks.
    - Open `Part2_Audio_Fingerprinting_and_Bonus.ipynb` for Fingerprinting.

3.  **Run All Cells:**
    The notebooks are self-contained. Ensure your `dataset/` and `songs/` folders are populated before running.

---

## 📊 Key Results

### Machine Learning
- **Genre Classification**: The Random Forest model achieves an F1-Micro score of **~0.44**. The confusion matrix reveals high overlap between similar genres (e.g., Pop/Dance).
- **Popularity**: Predicting popularity is challenging ($R^2 \approx 0.21$), indicating that external factors (marketing, trends) matter more than audio features.

### Audio Fingerprinting
- **Robustness**: The hashing algorithm successfully identifies songs even from short excerpts.
- **Search**: The system correctly matched queries (e.g., "Carmen Prelude") against the indexed database.

### Uncertainty Quantification
- **Conformal Prediction**: Demonstrated the trade-off between confidence and set size.
    - **95% Confidence**: Requires a set of ~16 genre candidates.
    - **80% Confidence**: Reduces to ~8 genre candidates.

---

## 📝 Report
A detailed technical report is available in the `repport/` directory:
- [Main.pdf](repport/Main.pdf) (Generated from LaTeX)

## 👥 Authors
- **AARES Project Team**
- IMT Nord Europe

---
*Built with ❤️ for Sound & Data.*
