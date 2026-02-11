import numpy as np
import librosa
import numba
from numba import jit
from utils_projet import get_maxima

def compute_spectrogram(x, sr=3000):
    """
    Computes the spectrogram of the signal x using librosa.stft.
    Parameters:
        x: Audio signal
        sr: Sampling rate
    Returns:
        S: Spectrogram magnitude
    """
    # STFT parameters from assignment
    n_fft = 2048
    hop_length = 512
    win_length = 1024
    
    # Compute STFT
    D = librosa.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Compute squared magnitude (Energy) as per Appendix B formula |Vg(f)(u,v)|^2
    S = np.abs(D)**2
    
    return S

@jit(nopython=True)
def get_maxima_in_tz(S, maxima, anchor):
    """
    Finds maxima in the Target Zone of an anchor point.
    Target Zone: 300 pixels in time and 20 pixels in frequency.
    Returns:
        neighbors: arrays of [freq, time] for the top 10 maxima in TZ
    """
    # Unpack anchor (frequency, time) - assuming maxima are [freq, time] from utils_projet
    f_anchor, t_anchor = anchor[0], anchor[1]
    
    # Define Target Zone boundaries relative to anchor
    # "not far away from it" as shown in Figure 1. 
    # Usually TZ starts a bit after anchor to avoid self-pairing and immediate neighbors.
    # Let's assume a small delay, e.g., 1 pixel, or start immediately? 
    # Fig 1 shows TZ is a box to the right. 
    # Let's define TZ as:
    # Time: [t_anchor + delta_t_min, t_anchor + delta_t_max]
    # Freq: [f_anchor - delta_f, f_anchor + delta_f]
    
    # Prompt: "find a Target Zone of 300 pixels in time and 20 pixels in frequency"
    # It doesn't specify offsets, but standard Shazam is forward in time.
    # Let's assume:
    # Time range: [t_anchor + 1, t_anchor + 300]
    # Freq range: [f_anchor - 10, f_anchor + 10] (total 20 height centered?) or [f_anchor - 20, f_anchor + 20]?
    # "20 pixels in frequency" likely means a band of height 20. 
    # Let's try [f_anchor - 10, f_anchor + 10].
    
    t_min = t_anchor + 10 # Small gap
    t_max = t_anchor + 300
    f_min = max(0, f_anchor - 10)
    f_max = min(S.shape[0], f_anchor + 10)
    
    # 1. Count candidates in TZ
    count = 0
    for i in range(len(maxima)):
        f = maxima[i][0]
        t = maxima[i][1]
        
        # Check if inside TZ
        if (t >= t_min) and (t <= t_max) and (f >= f_min) and (f <= f_max):
            count += 1
            
    if count == 0:
        return np.empty((0, 2), dtype=np.int64)
        
    # 2. Fill candidates
    candidate_indices = np.empty(count, dtype=np.int64)
    current_idx = 0
    
    for i in range(len(maxima)):
        f = maxima[i][0]
        t = maxima[i][1]
        
        # Check if inside TZ
        if (t >= t_min) and (t <= t_max) and (f >= f_min) and (f <= f_max):
            candidate_indices[current_idx] = i
            current_idx += 1
            
    # 3. Get amplitudes and sort
    candidate_amps = np.empty(count, dtype=np.float64)
    candidate_coords = np.empty((count, 2), dtype=np.int64)
    
    for k in range(count):
        idx = candidate_indices[k]
        f = int(maxima[idx][0])
        t = int(maxima[idx][1])
        candidate_amps[k] = S[f, t]
        candidate_coords[k, 0] = f
        candidate_coords[k, 1] = t
        
    # Sort by amplitude descending
    order = np.argsort(candidate_amps)[::-1]
    
    # Take top 10
    n_top = min(10, count)
    top_indices = order[:n_top]
    
    result = np.empty((n_top, 2), dtype=np.int64)
    for k in range(n_top):
        result[k] = candidate_coords[top_indices[k]]
        
    return result

def get_hashes(anchor, maxima_in_tz):
    """
    Generates hashes for pairs of anchor and target points.
    Hash formula: anchor[0]*1000000 + point[0]*1000 + point[1]-anchor[1]
    Returns list of (hash, time_offset) tuples.
    """
    hashes = []
    f_anchor, t_anchor = int(anchor[0]), int(anchor[1])
    
    for point in maxima_in_tz:
        f_point, t_point = int(point[0]), int(point[1])
        
        # Calculate hash
        # Formula: freq_anchor * 1Mz + freq_point * 1000 + delta_time
        # Check if delta_time is point[1] - anchor[1]
        delta_t = t_point - t_anchor
        
        hash_val = f_anchor * 1000000 + f_point * 1000 + delta_t
        
        # Return hash and absolute time of anchor (as per prompt/standard shazam)
        # Prompt: "return hash, anchor[1]"
        hashes.append((hash_val, t_anchor))
        
    return hashes

def process_signal(x, sr=3000):
    """
    Full pipeline: Spectrogram -> Maxima -> Hashes
    Returns: Dictionary of {hash: time_offset}
    """
    # 1. Spectrogram
    S = compute_spectrogram(x, sr=sr)
    
    # 2. Get Maxima
    # ensure generic python types for numba generic function if needed, 
    # but utils_projet.get_maxima uses numba so it expects numpy array
    maxima = get_maxima(S) # Returns [freq, time] pairs
    
    hashes_dict = {}
    
    # 3. Iterate anchors
    for i in range(len(maxima)):
        anchor = maxima[i]
        
        # 4. Get Target Zone Maxima
        maxima_in_tz = get_maxima_in_tz(S, maxima, anchor)
        
        # 5. Generate Hashes
        new_hashes = get_hashes(anchor, maxima_in_tz)
        
        # 6. Save to dictionary
        # Prompt says: "save all the hashes corresponding to the song as hash: anchor[1]"
        # Note: Collisions? 
        # "In a Python dictionary, save all the hashes... Return the dictionary"
        # If hash collision, we might overwrite. 
        # Standard implementation uses {hash: [list of times]}, but prompt implies simple dict.
        # "hash : anchor[1]" -> Single value. We'll overwrite or assume uniqueness enough.
        
        for h, t in new_hashes:
            hashes_dict[h] = t
            
    return hashes_dict

if __name__ == "__main__":
    # Test stub
    try:
        # Generate dummy signal
        sr = 3000
        t = np.linspace(0, 1, sr)
        x = np.sin(2*np.pi*440*t) + np.sin(2*np.pi*880*t)
        
        print("Processing dummy signal...")
        hashes = process_signal(x, sr=sr)
        print(f"Generated {len(hashes)} hashes.")
        
    except Exception as e:
        print(f"Error: {e}")
