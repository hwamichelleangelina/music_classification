import os
import librosa
import numpy as np
import pickle
import scipy.signal
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import librosa.util

data_path = "./data/original_data"
error_files = []

for root, _, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                y, sr = librosa.load(file_path, sr=22050)
            except Exception as e:
                error_files.append((file_path, str(e)))

# Hapus error
for file_path, error in error_files:
    print(f"Menghapus file error: {file_path} -> {error}")
    os.remove(file_path)

valid_files = []
for root, _, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            valid_files.append(os.path.join(root, file))

print(f"Total file yang masih valid: {len(valid_files)}")

def extract_features(y, sr, max_pad_len=40):
    """Ekstraksi fitur dari file audio."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    
    features = np.vstack([mfcc, chroma, spec_contrast, spectral_rolloff])
    features = librosa.util.normalize(features)
    
    pad_width = max_pad_len - features.shape[1]
    if pad_width > 0:
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_pad_len]
    
    return features

def augment_audio(y, sr):
    """Melakukan augmentasi audio."""
    augmented_data = []
    
    noise = 0.005 * np.random.randn(len(y))
    augmented_data.append(y + noise)
    
    n_steps = np.random.randint(-2, 3)
    if n_steps != 0:
        augmented_data.append(librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps))
    
    rate = np.random.uniform(0.8, 1.2)
    if rate != 1.0:
        augmented_data.append(librosa.effects.time_stretch(y, rate=rate))
    
    impulse_response = np.zeros(sr)
    impulse_response[0] = 1
    impulse_response[int(sr * 0.3)] = 0.6
    impulse_response[int(sr * 0.6)] = 0.3
    
    y_reverb = scipy.signal.fftconvolve(y, impulse_response, mode='full')[:len(y)]
    augmented_data.append(y_reverb)
    
    shift = np.random.randint(sr // 10, sr // 2)
    augmented_data.append(np.roll(y, shift))
    
    return augmented_data

def preprocess_data(data_path, save_path="data/original_data"):
    """Melakukan preprocessing pada dataset dan menyimpannya."""
    X, y = [], []
    genres = sorted(os.listdir(data_path))
    genre_labels = {genre: idx for idx, genre in enumerate(genres)}
    
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(genre_path, file)
                    try:
                        y_raw, sr = librosa.load(file_path, sr=22050)
                        features = extract_features(y_raw, sr)
                        X.append(features)
                        y.append(genre_labels[genre])
                        
                        for audio in augment_audio(y_raw, sr):
                            X.append(extract_features(audio, sr))
                            y.append(genre_labels[genre])
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    np.save(os.path.join(save_path, "X_features.npy"), X)
    np.save(os.path.join(save_path, "y_labels.npy"), y)
    
    with open(os.path.join(save_path, "dataset.pkl"), "wb") as f:
        pickle.dump((X, y), f)
    
    print(f"Dataset preprocessed. Shape: X={X.shape}, y={y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    
    return X, y