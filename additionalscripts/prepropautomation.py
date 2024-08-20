import os
import shutil
import librosa
import soundfile as sf

def preprocess_audio_data(raw_data_dir, preprocessed_data_dir):
    """
    Converts audio files to WAV format, renames them based on directory structure,
    and organizes them into emotion-based folders in a new preprocessed directory.

    Args:
        raw_data_dir: The path to the parent raw data directory containing researcher, gender, and emotion subfolders.
        preprocessed_data_dir: The path to the parent preprocessed data directory where emotion folders will be created.
    """

    # Clear existing preprocessed data directory and recreate it
    if os.path.exists(preprocessed_data_dir):
        shutil.rmtree(preprocessed_data_dir)
    os.makedirs(preprocessed_data_dir)

    # Create emotion folders in the preprocessed data directory
    emotion_folders = ['sad', 'surprised', 'happy', 'angry', 'calm']
    for emotion in emotion_folders:
        os.makedirs(os.path.join(preprocessed_data_dir, emotion))

    # Initialize file counter for sequential numbering
    file_counter = 1

    # Traverse the raw data directory and its subdirectories
    for root, dirs, files in os.walk(raw_data_dir):
        for filename in files:
            if filename.endswith(('.mp3', '.wav', '.flac', '.ogg')):  # Process supported audio formats
                input_path = os.path.join(root, filename)

                # Extract researcher, gender, and emotion from directory structure
                researcher_name = os.path.basename(os.path.dirname(os.path.dirname(root)))
                gender = os.path.basename(os.path.dirname(root)).lower()
                emotion_type = os.path.basename(root).lower()

                # Construct new filename using extracted information and file counter
                new_filename = f"{researcher_name[0]}{gender[0]}{emotion_type[0]}-{file_counter:03d}.wav"
                output_path = os.path.join(preprocessed_data_dir, emotion_type, new_filename)

                if not filename.endswith('.wav'):  # Convert to WAV if not already in that format
                    try:
                        y, sr = librosa.load(input_path)
                        sf.write(output_path, y, sr, subtype='PCM_24')
                        print(f"Converted and copied: {filename} to {output_path}") 
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                else:  # Copy the file if it's already in WAV format
                    shutil.copy2(input_path, output_path)
                    print(f"Copied: {filename} to {output_path}")

                file_counter += 1  # Increment the file counter for the next file

if __name__ == "__main__":
    raw_data_dir = input("Enter the parent raw data directory: ")
    preprocessed_data_dir = input("Enter the parent preprocessed data directory: ")

    preprocess_audio_data(raw_data_dir, preprocessed_data_dir)
    print("Preprocessing complete!")


# T:\Moringa\Data Science\v\audiofiles

# T:\Moringa\Data Science\v\capstone-project-ser\data

# T:\Moringa\Data Science\v\Swahili-Speech-Emotion-Recognition-System\data