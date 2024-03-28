import numpy as np
import os
import librosa
import noisereduce as nr
import tensorflow as tf
import json
from pydub import AudioSegment, effects
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define a function to find emotion from folder name
def find_emotion_from_folder_name(folder_name):
    if 'Bathroom_sink' in folder_name:
        return 0  # neutral
    elif 'Brushing_teeth' in folder_name:
        return 1  # calm
    elif 'Door' in folder_name:
        return 2  # happy
    elif 'Shower' in folder_name:
        return 3  # sad
    elif 'Sink' in folder_name:
        return 4  # angry
    elif 'Toilet' in folder_name:
        return 5  # fear
    elif 'Walking' in folder_name:
        return 6  # disgust
    else:
        return 7  # surprised

# Load the pre-trained model
saved_model_path = '/Users/priscilla/Developer/Speech-emotion-recognition/model8723.json'
saved_weights_path = '/Users/priscilla/Developer/Speech-emotion-recognition/model8723_weights.h5'

with open(saved_model_path, 'r') as json_file:
    json_savedModel = json_file.read()

# Loading the model architecture, weights
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(saved_weights_path)

# Define the folder path
folder_path = '/Users/priscilla/Developer/background_noise'

# Initialize lists to store true and predicted emotions
true_emotions = []
predicted_emotions = []

# Iterate through each folder
for folder_name in os.listdir(folder_path):
    folder_dir = os.path.join(folder_path, folder_name)
    # Skip if it's not a directory
    if not os.path.isdir(folder_dir):
        continue
    # Iterate through each audio file in the folder
    for file_name in os.listdir(folder_dir):
        try:
            # Load audio file
            audio_path = os.path.join(folder_dir, file_name)
            rawsound = AudioSegment.from_file(audio_path)
            normalizedsound = effects.normalize(rawsound, headroom=0)
            normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
            sr = librosa.get_samplerate(audio_path)
            xt, _ = librosa.effects.trim(normal_x, top_db=30)
            padded_x = np.pad(xt, (0, 173056 - len(xt)), 'constant')
            final_x = nr.reduce_noise(padded_x, sr=sr)

            # Extract features
            f1 = librosa.feature.rms(y=final_x, frame_length=2048, hop_length=512)
            f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=2048, hop_length=512, center=True)
            f3 = librosa.feature.mfcc(y=final_x, sr=sr, n_mfcc=13, hop_length=512)

            # Concatenate features
            X = np.concatenate((f1.T, f2.T, f3.T), axis=1)

            # Predict emotion
            prediction = model.predict(X.reshape(1, X.shape[0], X.shape[1]))

            # Append true and predicted emotions
            true_emotions.append(find_emotion_from_folder_name(folder_name))
            predicted_emotions.append(np.argmax(prediction))
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

# Generate confusion matrix
conf_matrix = confusion_matrix(true_emotions, predicted_emotions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["neutral", "happy", "sad", "angry", "fear", "disgust", "surprised"],
            yticklabels=["Bathroom_sink", "Brushing_teeth", "Door", "Shower", "Sink", "Toilet", "Walking"])
plt.xlabel('Predicted Emotion')
plt.ylabel('background_noise')
plt.title('Predicted Emotions of Background Sounds')
plt.show()
 
