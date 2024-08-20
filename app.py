import streamlit as st
import librosa
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import soundfile as sf

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load the models
rf_model = pickle.load(open("models/random_forest.pkl", "rb"))
xgb_model = pickle.load(open("models/xgboost.pkl", "rb"))
cat_model = pickle.load(open("models/catboost.pkl", "rb"))

models = {
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
    "CatBoost": cat_model
}

# Convert audio files to WAV format if necessary
def convert_to_wav(filepath):
    if not filepath.endswith('.wav'):
        data, samplerate = sf.read(filepath)
        new_filepath = filepath.rsplit('.', 1)[0] + '.wav'
        sf.write(new_filepath, data, samplerate)
        return new_filepath
    return filepath

# Function to extract features (similar to your FeatureExtractor class)
def extract_features(audio, sample_rate):
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

# Analyze a single audio file
def analyze_single_audio(audio):
    audio = convert_to_wav(audio)  # Convert to WAV if necessary
    audio, sample_rate = librosa.load(audio, sr=None)
    features = extract_features(audio, sample_rate)
    predictions = {model_name: model.predict_proba(features)[0] for model_name, model in models.items()}
    return predictions

# Analyze a folder of audio files
def analyze_folder(folder):
    summary = {model_name: np.zeros(len(models[model_name].classes_)) for model_name in models.keys()}
    files = [os.path.join(folder, f) for f in os.listdir(folder)]
    for file in files:
        file = convert_to_wav(file)  # Convert to WAV if necessary
        audio, sample_rate = librosa.load(file, sr=None)
        features = extract_features(audio, sample_rate)
        for model_name, model in models.items():
            summary[model_name] += model.predict_proba(features)[0]
    summary = {model_name: probs / len(files) for model_name, probs in summary.items()}
    return summary

# Analyze long audio files with segmentation
def analyze_long_audio(audio):
    audio = convert_to_wav(audio)  # Convert to WAV if necessary
    audio, sample_rate = librosa.load(audio, sr=None)
    segments = []
    duration = librosa.get_duration(y=audio, sr=sample_rate)
    for i in range(0, len(audio), 5 * sample_rate):
        segment = audio[i:i + 5 * sample_rate]
        if len(segment) < 1 * sample_rate:
            break
        features = extract_features(segment, sample_rate)
        segment_pred = {model_name: model.predict_proba(features)[0] for model_name, model in models.items()}
        segments.append(segment_pred)

    # Create a plot for the segment analysis
    time_points = np.arange(0, duration, 5)
    plt.figure(figsize=(10, 6))
    for model_name in models.keys():
        plt.plot(time_points[:len(segments)], [seg[model_name][1] for seg in segments], label=model_name)
    plt.xlabel('Time (s)')
    plt.ylabel('Probability')
    plt.title('Emotion Probability Over Time')
    plt.legend()
    plt.grid(True)
    
    return segments, plt

# Streamlit Interface
st.title("Skiza-AI: Emotion Analysis from Audio")

# Tab: Single Audio Analysis
st.header("Single Audio Analysis")
audio_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3", "ogg", "flac"])

if audio_file is not None:
    # Save the uploaded file temporarily
    with open("temp_audio_file", "wb") as f:
        f.write(audio_file.getbuffer())
    
    predictions = analyze_single_audio("temp_audio_file")
    st.subheader("Predicted Emotion Probabilities")
    st.json(predictions)

# Tab: Folder Analysis
st.header("Folder Analysis")
folder_path = st.text_input("Enter the path to a folder containing audio files:")

if folder_path:
    summary = analyze_folder(folder_path)
    st.subheader("Folder Summary of Predicted Emotion Probabilities")
    st.json(summary)

# Tab: Long Audio Analysis
st.header("Long Audio Analysis")
long_audio_file = st.file_uploader("Upload a Long Audio File", type=["wav", "mp3", "ogg", "flac"], key="long_audio")

if long_audio_file is not None:
    # Save the uploaded file temporarily
    with open("temp_long_audio_file", "wb") as f:
        f.write(long_audio_file.getbuffer())
    
    segments, plt_fig = analyze_long_audio("temp_long_audio_file")
    st.subheader("Segmented Emotion Analysis Over Time")
    st.pyplot(plt_fig)
