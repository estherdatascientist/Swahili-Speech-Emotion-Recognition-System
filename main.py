# Operating System and File Handling
import os

# Numerical Computation
import numpy as np

# Audio Processing
import librosa
import noisereduce as nr

# Machine Learning (scikit-learn)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize


from itertools import cycle

# Machine Learning (XGBoost and CatBoost)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Deep Learning (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Data Analysis and Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

class AudioPreprocessor(DataCleaner):
    def __init__(self, data_dir, emotions, sample_rate, target_length=16000, verbose=True):
        # Initialize with data directory, emotions, and target sample rate
        self.data_dir = data_dir
        self.emotions = emotions
        self.target_length = target_length  # Target length for padding/truncating audio files
        self.verbose = verbose  # Add a verbose flag

        # Initialize DataLoader to load and clean the data
        self.X = []
        self.y = []
        self.sample_rate = sample_rate

        self.load_data()
        self.X = self.clean_data()  # Clean the loaded data

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
            features = []

            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            mfccs_mean = np.mean(mfccs, axis=1)
            features.extend(mfccs_mean)

            # Extract Chroma
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)

            # Extract Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
            features.extend(spectral_contrast_mean)

            # Extract Zero Crossing Rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
            zero_crossing_rate_mean = np.mean(zero_crossing_rate)
            features.append(zero_crossing_rate_mean)

            # Extract Root Mean Square Energy
            rms = librosa.feature.rms(y=audio)
            rms_mean = np.mean(rms)
            features.append(rms_mean)

            # Extract Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
            spectral_centroid_mean = np.mean(spectral_centroid)
            features.append(spectral_centroid_mean)

            # Extract Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            features.append(spectral_bandwidth_mean)

            # Extract Spectral Roll-off
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            features.append(spectral_rolloff_mean)

            extracted_features.append(features)

        self.features = np.array(extracted_features)
        return self.features

    def get_features_and_labels(self):
        """
        Returns extracted features and corresponding numerical labels.
        """
        self.extract_features()
        return self.features, self.y

    

class EmotionLabeler(FeatureExtractor):
    def __init__(self, data_dir, emotions, sample_rate, target_length=16000, n_mfcc=13, verbose=True):
        super().__init__(data_dir, emotions, sample_rate, target_length, n_mfcc, verbose)
        self.emotion_map = {i: emotion for i, emotion in enumerate(emotions)}

    def label_emotions(self):
        labeled_emotions = [self.emotion_map[label] for label in self.y]
        return labeled_emotions

    def get_numerical_labels(self):
        return self.y


class DataSaver(EmotionLabeler):
    def __init__(self, data_dir, emotions, sample_rate, target_length=16000, n_mfcc=13, save_path="processed_data.csv", verbose=True):
        super().__init__(data_dir, emotions, sample_rate, target_length, n_mfcc, verbose)
        self.save_path = save_path

    def save_to_csv(self):
        features, labels = self.get_features_and_labels()
        df = pd.DataFrame(features)
        df['emotion'] = labels  # Save numerical labels instead of words
        df.to_csv(self.save_path, index=False)
        print(f"Data saved to {self.save_path}")

    # Remove `save_to_npy` method since it's not needed
    # def save_to_npy(self):
    #     features, labels = self.get_features_and_labels()
    #     np.save(self.save_path.replace('.csv', '_features.npy'), features)
    #     np.save(self.save_path.replace('.csv', '_labels.npy'), labels)
    #     print(f"Features and labels saved to {self.save_path.replace('.csv', '_features.npy')} and {self.save_path.replace('.csv', '_labels.npy')}")

    def split_data(self):
        features, labels = self.get_features_and_labels()
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
class Modeling:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        model = Sequential()
        
        # Define the input layer using the Input class
        model.add(Input(shape=self.input_shape))
        
        # Add LSTM layers
        model.add(LSTM(128, return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(LSTM(64, return_sequences=False))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        # Dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))

        self.model = model

    def compile_model(self, learning_rate=0.001):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def get_model(self):
        return self.model
    
class TrainingWithCallbacks(Modeling):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        # Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Model Checkpoint (with the updated .keras extension)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

        # Train the model
        history = self.model.fit(X_train, y_train,
                                 validation_data=(X_val, y_val),
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=[early_stopping, model_checkpoint])
        return history
    
class Evaluation(TrainingWithCallbacks):
    def __init__(self, input_shape, num_classes):
        super().__init__(input_shape, num_classes)

    def evaluate_model(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {results[0]}")
        print(f"Test Accuracy: {results[1]}")
        return results

    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def plot_confusion_matrix(self, X_test, y_test):
        # Predict the labels for the test set
        y_pred = np.argmax(self.model.predict(X_test), axis=1)

        # Compute the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

        # Print classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

class ModelSaver(Evaluation):
    def __init__(self, input_shape, num_classes, save_path='final_model.h5'):
        super().__init__(input_shape, num_classes)
        self.save_path = save_path

    def save_model(self):
        self.model.save(self.save_path)
        print(f"Model saved to {self.save_path}")
