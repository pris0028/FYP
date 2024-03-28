import numpy as np
import os
import random
from json_tricks import dump, load

from pydub import AudioSegment, effects
import librosa
import noisereduce as nr

import tensorflow as tf
import keras
import sklearn
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

import time
import audioread
import itertools

from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks

import matplotlib.pyplot as plt

## initialisation

# Emotion kind validation function for TESS database, due to emotions written within the file names.
def find_emotion_T(name): 
        if('neutral' in name): return "01"
        # elif('calm' in name): return "02"
        elif('happy' in name): return "03"
        elif('sad' in name): return "04"
        elif('angry' in name): return "05"
        elif('fear' in name): return "06"
        elif('disgust' in name): return "07"
        elif('suprised' in name): return "08"
        else: return "-1"
        
        
# 'emotions' list fix for classification purposes:
#     Classification values start from 0, Thus an 'n = n-1' operation has been executed for both RAVDESS and TESS databases:
def emotionfix(e_num):
    if e_num == "01":   return 0 # neutral
#     elif e_num == "02": return 1 # calm
    elif e_num == "03": return 2 # happy
    elif e_num == "04": return 3 # sad
    elif e_num == "05": return 4 # angry
    elif e_num == "06": return 5 # fear
    elif e_num == "07": return 6 # disgust
    else:               return 7 # suprised


# Set the audioread backend explicitly
audioread.ffdec.FFmpegAudioFile.default_ffmpeg_exe = 'path_to_ffmpeg'

## preprocessing

tic = time.perf_counter()

# Initialize data lists
rms = []
zcr = []
mfcc = []
emotions = []

# Initialize variables
total_length = 173056  # desired frame length for all of the audio samples.
frame_length = 2048
hop_length = 512

# Background noise paths
noise_paths = [
    "/Users/priscilla/Developer/background_noise/Bathroom_sink",
    "/Users/priscilla/Developer/background_noise/Brushing_teeth",
    "/Users/priscilla/Developer/background_noise/Door",
    "/Users/priscilla/Developer/background_noise/Shower",
    "/Users/priscilla/Developer/background_noise/Sink",
    "/Users/priscilla/Developer/background_noise/Toilet",
    "/Users/priscilla/Developer/background_noise/Walking"
]

# Original train data path
train_path = "/Users/priscilla/Developer/Speech-emotion-recognition/audio_files"

for subdir, dirs, files in os.walk(train_path):
    for file in files:
        try:
            # Fetch the sample rate.
            _, sr = librosa.load(path=os.path.join(subdir, file), sr=None)
            # Load the audio file using pydub.
            rawsound = AudioSegment.from_file(os.path.join(subdir, file))
            
            # Randomly select background noise and superimpose
            noise_path = random.choice(noise_paths)
            noise_file = random.choice(os.listdir(noise_path))
            noise_audio = AudioSegment.from_file(os.path.join(noise_path, noise_file))
            noise_start = random.randint(0, len(noise_audio) - len(rawsound))
            noise_end = noise_start + len(rawsound)
            noise_segment = noise_audio[noise_start:noise_end]
            combined = rawsound.overlay(noise_segment)
            
            # Normalize the audio to +5.0 dBFS.
            normalizedsound = effects.normalize(combined, headroom=0)
            # Transform the normalized audio to np.array of samples.
            normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
            # Trim silence from the beginning and the end.
            xt, index = librosa.effects.trim(normal_x, top_db=30)
            # Pad for duration equalization.
            padded_x = np.pad(xt, (0, total_length - len(xt)), 'constant')
            # Noise reduction.
            final_x = nr.reduce_noise(padded_x, sr=sr)  # updated 03/03/22

            # Features extraction
            f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length)
            f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frame_length, hop_length=hop_length,
                                                    center=True)  # ZCR
            f3 = librosa.feature.mfcc(y=final_x, sr=sr, n_mfcc=13, hop_length=hop_length)

            # Emotion extraction from the different databases
            if (find_emotion_T(file) != "-1"):  # TESS database validation
                name = find_emotion_T(file)
            else:  # RAVDESS database validation
                name = file[6:8]

            # Filling the data lists
            rms.append(f1)
            zcr.append(f2)
            mfcc.append(f3)
            emotions.append(emotionfix(name))

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue  # Continue with the next file

toc = time.perf_counter()
print(f"Running time: {(toc - tic)/60:0.4f} minutes")

# Adjusting features shape to the 3D format: (batch, timesteps, feature)

f_rms = np.asarray(rms).astype('float32')
f_rms = np.swapaxes(f_rms,1,2)
f_zcr = np.asarray(zcr).astype('float32')
f_zcr = np.swapaxes(f_zcr,1,2)
f_mfccs = np.asarray(mfcc).astype('float32')
f_mfccs = np.swapaxes(f_mfccs,1,2)

print('ZCR shape:',f_zcr.shape)
print('RMS shape:',f_rms.shape)
print('MFCCs shape:',f_mfccs.shape)

# Concatenating all features to 'X' variable.
X = np.concatenate((f_zcr, f_rms, f_mfccs), axis=2)

# Preparing 'Y' as a 2D shaped variable.
Y = np.asarray(emotions).astype('int8')
Y = np.expand_dims(Y, axis=1)

##test
# Reading the model from JSON file

saved_model_path = '/Users/priscilla/Developer/Speech-emotion-recognition/model8723.json'
saved_weights_path = '/Users/priscilla/Developer/Speech-emotion-recognition/model8723_weights.h5'

with open(saved_model_path , 'r') as json_file:
    json_savedModel = json_file.read()
    
# Loading the model architecture, weights
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(saved_weights_path)

# Compiling the model with similar parameters as the original model.
model.compile(loss='categorical_crossentropy', 
                optimizer='RMSProp', 
                metrics=['categorical_accuracy'])

# # Model's structure visualization
# tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

predictions = model.predict(X)
y_pred_class = np.argmax(predictions, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(Y.flatten(), y_pred_class)

# Print classification report
report = classification_report(Y.flatten(), y_pred_class, output_dict=True)

# Mapping emotion indices to their corresponding labels
emotion_labels = {
    0: "neutral",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fear",
    6: "disgust",
    7: "suprised"
}

# Print classification report in a tabular format
print("\nClassification Report:")
print("{:<10} {:<10} {:<10} {:<10} {:<10}".format('Emotion', 'Precision', 'Recall', 'F1-Score', 'Support'))
print("-" * 50)
for emotion, label in emotion_labels.items():
    precision = report[str(emotion)]['precision']
    recall = report[str(emotion)]['recall']
    f1_score = report[str(emotion)]['f1-score']
    support = report[str(emotion)]['support']
    print("{:<10} {:<10.2f} {:<10.2f} {:<10.2f} {:<10.0f}".format(label, precision, recall, f1_score, support))

print("-" * 50)
print("Accuracy: {:.2f}".format(report['accuracy']))

# Print accuracy for each emotion
print("\nAccuracy for each emotion:")
for emotion, label in emotion_labels.items():
    accuracy = report[str(emotion)]['precision']
    print(f"{label}: {accuracy*100:.2f}%")

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(emotion_labels))
plt.xticks(tick_marks, emotion_labels.values(), rotation=45)
plt.yticks(tick_marks, emotion_labels.values())

fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# Outputting predicted emotions
predicted_emotions = [emotion_labels[idx] for idx in y_pred_class]
print(predicted_emotions)
