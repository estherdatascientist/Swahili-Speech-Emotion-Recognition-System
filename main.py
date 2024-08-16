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