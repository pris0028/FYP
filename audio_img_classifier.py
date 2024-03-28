import subprocess
import os
import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment
from emotion import emotionfix, find_emotion_T
import cv2
import time

# Function to record audio using arecord
def record_audio(duration, filename):
    print(f"Recording audio for {duration} seconds...")
    subprocess.run(['arecord', '-D', 'plughw:3,0', '-d', str(duration), filename])
    print("Recording finished.")

# Define the file path to save the recorded audio in WAV format
audio_file = '/home/pi/Developer/audio/test.wav'

# Record audio for 3 seconds
record_audio(3, audio_file)
time.sleep(3)

# Check if the audio file exists
if not os.path.exists(audio_file):
    print("Error: Audio file not found. Recording might have failed.")
    exit()

# Load the audio file using pydub
rawsound = AudioSegment.from_file(audio_file)
# Normalize the audio to +5.0 dBFS
normalizedsound = rawsound.apply_gain(5.0)

# Transform the normalized audio to np.array of samples
normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
# Trim silence from the beginning and the end
xt, index = librosa.effects.trim(normal_x, top_db=30)
# Pad for duration equalization
padded_x = np.pad(xt, (0, total_length - len(xt)), 'constant')
# Noise reduction
final_x = nr.reduce_noise(padded_x, sr=sr)

# Features extraction
f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length)
f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frame_length, hop_length=hop_length, center=True)  # ZCR
f3 = librosa.feature.mfcc(y=final_x, sr=sr, n_mfcc=13, hop_length=hop_length)

# Filling the data lists
rms.append(f1)
zcr.append(f2)
mfcc.append(f3)

# Emotion extraction from the different databases
if (find_emotion_T(file) != "-1"):  # TESS database validation
    name = find_emotion_T(file)
else:  # RAVDESS database validation
    name = file[6:8]
emotions.append(emotionfix(name))

# Adjusting features shape to the 3D format: (batch, timesteps, feature)
f_rms = np.asarray(rms).astype('float32')
f_rms = np.expand_dims(f_rms, axis=0)  # Add batch dimension
f_rms = np.swapaxes(f_rms, 1, 2)

f_zcr = np.asarray(zcr).astype('float32')
f_zcr = np.expand_dims(f_zcr, axis=0)  # Add batch dimension
f_zcr = np.swapaxes(f_zcr, 1, 2)

f_mfccs = np.asarray(mfcc).astype('float32')
f_mfccs = np.expand_dims(f_mfccs, axis=0)  # Add batch dimension
f_mfccs = np.swapaxes(f_mfccs, 1, 2)

# Concatenating all features to 'X' variable
X = np.concatenate((f_zcr, f_rms, f_mfccs), axis=2)

# Load model and weights
saved_model_path = '/home/pi/Developer/model8723.json'
saved_weights_path = '/home/pi/Developer/model8723_weights.h5'

with open(saved_model_path, 'r') as json_file:
    json_savedModel = json_file.read()

# Load the model architecture and weights
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(saved_weights_path)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_accuracy'])

# Make predictions
predictions = model.predict(X)
y_pred_class = np.argmax(predictions, axis=1)

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

# Determine if emotion is fearful
if "fear" in predicted_emotions:
    predicted_emotion = "fear"
else:
    predicted_emotion = "not_fear"

# Run thermal image classification
# Assuming the thermal images are already captured and stored in the directory

# Your code snippet to run thermal image classification goes here
# Assuming you already have a function to compare temperatures, similar to the one in thermalimg.py
# Load the thermal image
def calculate_mean_temperature(image_path):
    # Read the thermal image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Split the image into top and bottom halves
    height, width = img.shape
    top_half = img[:height//2, :]
    bottom_half = img[height//2:, :]
    
    # Calculate mean temperature of each half
    top_mean_temp = np.mean(top_half)
    bottom_mean_temp = np.mean(bottom_half)
    
    return top_mean_temp, bottom_mean_temp

def compare_temperature(image_path):
    top_mean_temp, bottom_mean_temp = calculate_mean_temperature(image_path)
    
    # Determine which half is warmer
    if top_mean_temp > bottom_mean_temp:
        print("Top half is warmer")
        return "Top half is warmer"
    elif top_mean_temp < bottom_mean_temp:
        print("Bottom half is warmer")
        return "Bottom half is warmer"
    else:
        print("Both halves have similar temperatures")
        return "Both halves have similar temperatures"

# Placeholder result for thermal image classification
thermal_result = compare_temperature(image_path)  # Replace 'image_path' with the path to your thermal image

# Buzz the buzzer if emotion is fearful or bottom half of the thermal image is warmer
if predicted_emotion == "fear" or thermal_result == "Bottom half is warmer":
    # Code to buzz the buzzer goes here
    print("Buzz the buzzer!")
else:
    print("No action needed.")
