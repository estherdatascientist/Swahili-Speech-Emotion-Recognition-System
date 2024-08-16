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
