import os
import librosa
import numpy as np
import pickle
import soundfile as sf
import logging
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import uuid  # For generating unique temporary filenames

# Load models
rf_model = pickle.load(open("models/random_forest.pkl", "rb"))
xgb_model = pickle.load(open("models/xgboost.pkl", "rb"))
cat_model = pickle.load(open("models/catboost.pkl", "rb"))

models = {
    "CatBoost (Recommended)": cat_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

emotion_map = {
    0: 'sad',
    1: 'happy',
    2: 'surprised',
    3: 'angry',
    4: 'calm'
}
emotion_labels = list(emotion_map.values())

def convert_to_wav(audio_file):
    try:
        data, samplerate = sf.read(audio_file)
        new_filepath = f"temp_{uuid.uuid4().hex}.wav"  # Generate a unique filename
        sf.write(new_filepath, data, samplerate)
        return new_filepath
    except Exception as e:
        logging.error(f"Error converting {audio_file} to WAV: {e}")
        return None

def extract_features(audio, sample_rate):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1),
            np.mean(librosa.feature.zero_crossing_rate(y=audio)),
            np.mean(librosa.feature.rms(y=audio)),
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)),
            np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)),
            np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
        ])
        return features.reshape(1, -1)
    except Exception as e:
        logging.error(f"Error extracting features: {e}")
        return None

def analyze_folder(folder_path):
    summary = {emotion: [] for emotion in emotion_labels}
    
    # Log folder path and files
    logging.info(f"Analyzing folder: {folder_path}")
    
    try:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
    except Exception as e:
        logging.error(f"Error listing files in folder: {e}")
        return {}
    
    for file_path in files:
        logging.info(f"Processing file: {file_path}")
        
        if os.path.isfile(file_path):
            try:
                wav_file_path = convert_to_wav(file_path)
                if wav_file_path is None:
                    continue
                
                audio, sample_rate = librosa.load(wav_file_path, sr=None)
                features = extract_features(audio, sample_rate)
                
                if features is None:
                    continue
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue

            try:
                for model_name, model in models.items():
                    predictions = model.predict_proba(features)[0]
                    for emotion, prob in zip(emotion_labels, predictions):
                        summary[emotion].append(prob)
            except Exception as e:
                logging.error(f"Error making predictions for file {file_path}: {e}")
                continue

            # Clean up temporary WAV file
            try:
                os.remove(wav_file_path)
            except Exception as e:
                logging.error(f"Error deleting temporary file {wav_file_path}: {e}")
    
    avg_probs = {emotion: np.mean(probs) * 100 for emotion, probs in summary.items() if probs}
    return avg_probs
