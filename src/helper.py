import os
from collections import Counter, defaultdict
import librosa
import numpy as np

def get_subfolders(path):
    subfolders = []
    for root, dirs, _ in os.walk(path):
        for d in dirs:
            subfolders.append(os.path.relpath(os.path.join(root, d), path))
    return sorted(subfolders)

def get_file_stats(folder_path):
    file_prefixes = []
    file_extensions = []
    file_count = 0

    for root, _, files in os.walk(folder_path):
        for f in files:
            file_count += 1

            name, ext = os.path.splitext(f)
            prefix = ''.join(ch for ch in name if not ch.isdigit())
            ext = ext.lower().lstrip(".")

            file_prefixes.append(prefix if prefix else "NO_PREFIX")
            file_extensions.append(ext if ext else "NO_EXT")


    return file_count, Counter(file_prefixes), Counter(file_extensions)

def extract_audio_features(filepath):
    """Load audio and extract basic librosa features.
    
    - duration (seconds)
    - sample_rate (number of audio samples recorded per second)
    - rms_energy (average loudness)
    - zero_crossing_rate (how often the signal changes sign → rough measure of noisiness)
    - spectral_centroid (center of mass of spectrum → perceived brightness)
    
    """
    try:
        y, sr = librosa.load(filepath, sr=None)  # keep original sample rate
        duration = librosa.get_duration(y=y, sr=sr)
        
        rms = float(np.mean(librosa.feature.rms(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        
        return {
            "duration": duration,
            "sample_rate": sr,
            "rms_energy": rms,
            "zero_crossing_rate": zcr,
            "spectral_centroid": centroid
        }
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {
            "duration": None,
            "sample_rate": None,
            "rms_energy": None,
            "zero_crossing_rate": None,
            "spectral_centroid": None
        }