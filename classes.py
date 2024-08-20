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
import pickle

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
        """
        Initializes the EmotionLabeler, inheriting from FeatureExtractor.

        Args:
            data_dir (str): Path to the directory containing audio files.
            emotions (list): List of emotion labels.
            sample_rate (int): Target sample rate for audio processing.
            target_length (int): Desired length for audio padding/truncation.
            n_mfcc (int): Number of MFCC coefficients to extract.
            verbose (bool): Whether to print progress messages.
        """
        super().__init__(data_dir, emotions, sample_rate, target_length, n_mfcc, verbose)

        # Define the explicit mapping of emotions to numerical labels
        self.emotion_map = {
            'sad': 0,
            'happy': 1,
            'surprised': 2,
            'angry': 3,
            'calm': 4
        }

    def label_emotions(self):
        """
        Converts numerical labels back to their corresponding emotion names.

        Returns:
            list: A list of emotion labels corresponding to the numerical labels in self.y.
        """
        return [self.emotion_map[label] for label in self.y]

    def get_numerical_labels(self):
        """
        Returns the numerical labels.

        Returns:
            numpy.ndarray: The numerical labels (self.y).
        """
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

    def split_data(self):
        features, labels = self.get_features_and_labels()
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
class Modeling:
    def __init__(self, model_name, input_shape=None, num_classes=None):
        self.model_name = model_name
        self.input_shape = input_shape  # Only used for neural networks
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        if self.model_name == 'knn':
            self.model = KNeighborsClassifier()
        elif self.model_name == 'random_forest':
            self.model = RandomForestClassifier()
        elif self.model_name == 'svm':
            self.model = SVC(probability=True)
        elif self.model_name == 'xgboost':
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
        elif self.model_name == 'catboost':
            self.model = CatBoostClassifier(verbose=0)
        elif self.model_name == 'mlp':
            self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300)


class TrainingWithCallbacks(Modeling):
    def __init__(self, model_name, input_shape=None, num_classes=None):
        super().__init__(model_name, input_shape, num_classes)

    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        if self.model_name in ['knn', 'random_forest', 'svm', 'xgboost', 'catboost']:
            self.model.fit(X_train, y_train)
        elif self.model_name == 'mlp':
            # For MLP, we don't use epochs and batch_size directly
            self.model.fit(X_train, y_train)  # No need to pass epochs and batch_size
            return None  # No history object returned in scikit-learn models
        else:
            raise ValueError("Unsupported model name.")


class Evaluation(TrainingWithCallbacks):
    def __init__(self, model_name, input_shape=None, num_classes=None):
        super().__init__(model_name, input_shape, num_classes)
        self.history = None

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        
        # Handle predictions based on model type
        if self.model_name in ['knn', 'random_forest', 'svm', 'xgboost', 'catboost']:
            y_pred = y_pred  # Predicted labels are directly provided
        elif self.model_name == 'mlp':
            y_pred = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred
        else:
            raise ValueError("Unsupported model name.")

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy}")

        # Generate and display the confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        # Generate and print the classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Plot ROC Curve based on the number of classes
        if self.num_classes == 2:
            self.plot_roc_curve(y_test, y_pred)
        else:
            self.plot_multiclass_roc(y_test, y_pred)

        return accuracy

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_training_history(self):
        if self.history is None:
            print("No training history available for non-neural network models.")
            return

        plt.figure(figsize=(12, 4))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Val Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def plot_roc_curve(self, y_test, y_pred):
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

    def plot_multiclass_roc(self, y_test, y_pred):
        y_test_bin = label_binarize(y_test, classes=range(self.num_classes))
        y_pred_bin = label_binarize(y_pred, classes=range(self.num_classes))

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(8, 6))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(self.num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multiclass Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def plot_feature_importance(self):
        if self.model_name in ['random_forest', 'xgboost', 'catboost']:
            importance = self.model.feature_importances_
            plt.figure(figsize=(10, 7))
            plt.bar(range(len(importance)), importance)
            plt.title('Feature Importance')
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.show()
        else:
            print(f"Feature importance not available for {self.model_name}.")

    def evaluate_and_plot(self, X_test, y_test):
        # Evaluate the model and print classification report
        accuracy = self.evaluate_model(X_test, y_test)

        # Plot relevant plots
        self.plot_training_history()  # Only for neural network models
        self.plot_feature_importance()  # Only for models that support it

        return accuracy



class ModelSaver:
    def __init__(self, trained_models, save_dir="models"):
        """
        Initialize the ModelSaver class with trained models and a directory to save them.
        
        Parameters:
        - trained_models (dict): A dictionary where keys are model names and values are trained model instances.
        - save_dir (str): The directory where models will be saved. Defaults to 'models'.
        """
        self.trained_models = trained_models
        self.save_dir = save_dir

        # Create the save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Directory '{self.save_dir}' created.")
        else:
            print(f"Directory '{self.save_dir}' already exists.")

    def save_models(self):
        """
        Save the trained models to the specified directory as pickle files.
        """
        for model_name, model_instance in self.trained_models.items():
            model_file_path = os.path.join(self.save_dir, f"{model_name}.pkl")
            try:
                with open(model_file_path, 'wb') as model_file:
                    pickle.dump(model_instance, model_file)
                print(f"Model '{model_name}' saved successfully at '{model_file_path}'.")
            except Exception as e:
                print(f"Failed to save model '{model_name}': {e}")


