import os
import numpy as np
import librosa
import noisereduce as nr
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras.layers import Input, Dropout 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class DataLoader:
    def __init__(self, data_dir, emotions):
        self.data_dir = data_dir
        self.emotions = emotions
        self.X = []  # List to store audio data (as numpy arrays)
        self.y = []  # List to store corresponding labels

    def load_data(self):
        for i, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(self.data_dir, emotion)
            wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
            print(f"Number of .wav files in {emotion} folder: {len(wav_files)}")

            for filename in wav_files:
                filepath = os.path.join(emotion_dir, filename)
                audio, sample_rate = librosa.load(filepath)
                self.X.append(audio)
                self.y.append(i)  # Use emotion index as label

        self.y = np.array(self.y)  # Convert labels to numpy array
class DataCleaner(DataLoader):
    def __init__(self, X, sample_rate):
        self.X = X
        self.sample_rate = sample_rate

    def clean_data(self):
        cleaned_X = []
        for audio in self.X:
            # Noise Reduction
            cleaned_audio = nr.reduce_noise(y=audio, sr=self.sample_rate)

            # Trim Silence
            cleaned_audio, _ = librosa.effects.trim(cleaned_audio)

            cleaned_X.append(cleaned_audio)
        return cleaned_X
    
    def load_data(self):
            """
            Load audio files and store them in X.
            """
            for i, emotion in enumerate(self.emotions):
                emotion_dir = os.path.join(self.data_dir, emotion)
                wav_files = [f for f in os.listdir(emotion_dir) if f.endswith('.wav')]
                
                if self.verbose:
                    print(f"Processing {len(wav_files)} files for emotion: {emotion}")

                for filename in wav_files:
                    filepath = os.path.join(emotion_dir, filename)
                    try:
                        # Load the audio file
                        audio, _ = librosa.load(filepath, sr=self.sample_rate)
                        self.X.append(audio)
                        self.y.append(i)

                    except Exception as e:
                        if self.verbose:
                            print(f"Error processing file {filepath}: {e}")

            self.y = np.array(self.y)  # Convert labels to numpy array
    
    def pad_audio(self):
        """
        Pads or truncates the audio array to the target length.
        """
        padded_X = []
        for audio in self.X:
            if len(audio) > self.target_length:
                padded_audio = audio[:self.target_length]
            else:
                padded_audio = np.pad(audio, (0, self.target_length - len(audio)), 'constant')
            padded_X.append(padded_audio)

        return np.array(padded_X)

    def get_data(self):
        """
        Returns the processed and padded data and labels.
        """
        self.X = self.pad_audio()
        return np.array(self.X), np.array(self.y)
    
class FeatureExtractor(AudioPreprocessor):
    def __init__(self, data_dir, emotions, sample_rate, target_length=16000, n_mfcc=13, verbose=True):
        super().__init__(data_dir, emotions, sample_rate, target_length, verbose)
        self.n_mfcc = n_mfcc
        self.features = None  # To store the extracted features

    def extract_features(self):
        extracted_features = []
        for audio in self.X:
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfccs_scaled = np.mean(mfccs.T, axis=0)  # Take the mean across time steps
            extracted_features.append(mfccs_scaled)

        self.features = np.array(extracted_features)
        return self.features

    def get_features_and_labels(self):
        """
        Returns extracted features and corresponding numerical labels.
        """
        self.extract_features()
        return self.features, self.y
