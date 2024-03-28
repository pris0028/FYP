import numpy as np
import os
from json_tricks import dump, load

from pydub import AudioSegment, effects
import librosa
import noisereduce as nr

import tensorflow as tf
import keras
import sklearn
import pandas as pd

import time
import audioread

from keras.models import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks 

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
        elif('ps' in name): return "08"
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

# sample_lengths = []
# folder_path = '/Users/priscilla/Developer/Speech-emotion-recognition/audiofiles2'

# for subdir, dirs, files in os.walk(folder_path):
#     for file in files:
#         try:
#             file_path = os.path.join(subdir, file)
#             x, sr = librosa.load(file_path, sr=None, mono=True, dtype=np.float32)
#             xt, index = librosa.effects.trim(x, top_db=30)
#             sample_lengths.append(len(xt))
#         except Exception as e:
#             print(f"Error processing file {file_path}: {e}")

# if sample_lengths:
#     print('Maximum sample length:', max(sample_lengths))
# else:
#     print('No valid audio files found in the specified directory.')

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

folder_path = '/Users/priscilla/Developer/Speech-emotion-recognition/audiofiles2'

for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        try:
            # Fetch the sample rate.
            _, sr = librosa.load(path=os.path.join(subdir, file), sr=None)
            # Load the audio file using pydub.
            rawsound = AudioSegment.from_file(os.path.join(subdir, file))
            # Normalize the audio to +5.0 dBFS.
            normalizedsound = effects.normalize(rawsound, headroom=0)
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

# # Preparing 'Y' as a 2D shaped variable.
# Y = np.asarray(emotions).astype('int8')
# Y = np.expand_dims(Y, axis=1)

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
print(predictions,y_pred_class)

# Mapping emotion indices to their corresponding labels
emotion_labels = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fear",
    6: "disgust",
    7: "suprised"
}

# Outputting predicted emotions
predicted_emotions = [emotion_labels[idx] for idx in y_pred_class]
print(predicted_emotions)