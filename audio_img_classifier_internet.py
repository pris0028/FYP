import os
import wave
import librosa
import numpy as np
import noisereduce as nr
from pydub import AudioSegment, effects
from pydub.playback import play
from datetime import datetime
import tensorflow as tf
import requests
import urllib.parse
import time
import cv2
import json
import board
import busio
import adafruit_mlx90640
from PIL import Image
import matplotlib.pyplot as plt

# Load the emotion detection model
saved_model_path = '/home/pi/Developer/model8723.json'
saved_weights_path = '/home/pi/Developer/model8723_weights.h5'

with open(saved_model_path, 'r') as json_file:
    json_savedModel = json_file.read()

model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(saved_weights_path)
model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_crossentropy'])

# Emotion kind validation function for TESS database
def find_emotion_T(name):
    if ('neutral' in name):
        return "01"
    elif ('happy' in name):
        return "03"
    elif ('sad' in name):
        return "04"
    elif ('angry' in name):
        return "05"
    elif ('fear' in name):
        return "06"
    elif ('disgust' in name):
        return "07"
    elif ('ps' in name):
        return "08"
    else:
        return "-1"

# 'emotions' list fix for classification purposes
def emotionfix(e_num):
    if e_num == "01":
        return 0  # neutral
    elif e_num == "03":
        return 2  # happy
    elif e_num == "04":
        return 3  # sad
    elif e_num == "05":
        return 4  # angry
    elif e_num == "06":
        return 5  # fear
    elif e_num == "07":
        return 6  # disgust
    else:
        return 7  # suprised

# Record audio
def record_audio():
    directory = "/home/pi/Developer/audio"
    filename = os.path.join(directory, "recorded_audio.wav")
    cmd = f"arecord -D plughw:3,0 -d 3 -f S16_LE -r 16000 {filename}"
    os.system(cmd)
    return filename
    
def convert_png_to_bmp(png_path, bmp_path):
    # Open the PNG image
    png_image = Image.open(png_path)

    # Resize the image to 160x128 pixels (landscape)
    resized_image = png_image.resize((160, 128))

    # Convert the image to BMP format
    bmp_image = resized_image.convert("RGB")

    # Save the BMP image to the specified file path
    bmp_image.save(bmp_path)

    print(f"Conversion successful. BMP image saved to {bmp_path}")

# Capture thermal image
def capture_thermal_image():
    while True:
        try:
            i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
            mlx = adafruit_mlx90640.MLX90640(i2c)
            mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
            mlx_shape = (24, 32)

            frame = np.zeros((24 * 32,))
            mlx.getFrame(frame)
            data_array = np.reshape(frame, mlx_shape)

            #plt.figure(figsize=(8, 6)) #save thermal image without the heat axis
            plt.figure(figsize=(6,4.8)) #save the thermal image without the heat axis
            plt.imshow(data_array, cmap='hot', interpolation='bilinear')  # Using bilinear interpolation
            #plt.colorbar() # include the color bar at the side
            plt.axis('off')
            plt.savefig("/home/pi/Developer/mlx90640_test_fliplr.png", bbox_inches='tight', pad_inches=0)
            #plt.show() # show the image
            plt.close()
            
            # Convert PNG to BMP
            png_path = "/home/pi/Developer/mlx90640_test_fliplr.png"
            bmp_path = "/media/pi/CIRCUITPY/images/image.bmp"
            convert_png_to_bmp(png_path, bmp_path)
            
            break  # Break the loop if the image is successfully captured
        except Exception as e:
            print("An error occurred:", e)
            time.sleep(1)  # Wait for a second before trying again

# Process audio for emotion detection
def process_audio(audio_file):
    total_length = 173056  # desired frame length for all of the audio samples.
    frame_length = 2048
    hop_length = 512

    # Load the audio file using pydub.
    rawsound = AudioSegment.from_file(audio_file)
    # Normalize the audio to +5.0 dBFS.
    normalizedsound = effects.normalize(rawsound, headroom=0)
    # Transform the normalized audio to np.array of samples.
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype='float32')
    # Trim silence from the beginning and the end.
    xt, index = librosa.effects.trim(normal_x, top_db=30)
    # Pad for duration equalization.
    padded_x = np.pad(xt, (0, total_length - len(xt)), 'constant')
    # Noise reduction.
    final_x = nr.reduce_noise(padded_x, sr=16000)  # updated 03/03/22

    # Features extraction
    f1 = librosa.feature.rms(y=final_x, frame_length=frame_length, hop_length=hop_length)
    f2 = librosa.feature.zero_crossing_rate(final_x, frame_length=frame_length, hop_length=hop_length, center=True)  # ZCR
    f3 = librosa.feature.mfcc(y=final_x, sr=16000, n_mfcc=13, hop_length=hop_length)

    # Adjusting features shape to the 3D format: (batch, timesteps, feature)
    f_rms = np.asarray([f1]).astype('float32')
    f_rms = np.swapaxes(f_rms, 1, 2)
    f_zcr = np.asarray([f2]).astype('float32')
    f_zcr = np.swapaxes(f_zcr, 1, 2)
    f_mfccs = np.asarray([f3]).astype('float32')
    f_mfccs = np.swapaxes(f_mfccs, 1, 2)

    # Concatenating all features to 'X' variable.
    X = np.concatenate((f_zcr, f_rms, f_mfccs), axis=2)

    predictions = model.predict(X)
    y_pred_class = np.argmax(predictions, axis=1)

    emotion_labels = {
        0: "neutral",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fear",
        6: "disgust",
        7: "suprised"
    }

    predicted_emotions = [emotion_labels[idx] for idx in y_pred_class]
    detected_emotion = max(set(predicted_emotions), key=predicted_emotions.count)
    print("Detected Emotion:", detected_emotion)
    return predicted_emotions

# Process thermal image
def process_thermal_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read the thermal image.")
        return False
    height, width = img.shape
    top_half = img[:height // 2, :]
    bottom_half = img[height // 2:, :]

    top_mean_temp = np.mean(top_half)
    bottom_mean_temp = np.mean(bottom_half)

    if top_mean_temp > bottom_mean_temp:
        print("Top half is warmer")
        return False
    elif top_mean_temp < bottom_mean_temp:
        print("Bottom half is warmer")
        return True
    else:
        print("Top and bottom halves have equal temperature")
        return False

# Send telegram call and message
def send_telegram_alert():
    username = "imprisla"
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    message = "A fall has taken place at " + date_time
    encoded_message = urllib.parse.quote(message)

    # Make the voice call
    voice_url = f"https://api.callmebot.com/start.php?user=@{username}&text={encoded_message}"
    response = requests.get(voice_url)
    print(f"Voice call initiated: {response.status_code}")

def play_reassurance_audios(audio_folder):
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith(".mp3"):
            audio_path = os.path.join(audio_folder, audio_file)
            # Convert MP3 to WAV
            sound = AudioSegment.from_mp3(audio_path)
            # Play the audio
            play(sound)
            
def send_text_message():
    username = "imprisla"
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    message = "A fall has taken place at " + date_time
    encoded_message = urllib.parse.quote(message)

    # Send the text message
    text_url = f"https://api.callmebot.com/text.php?user=@{username}&text={encoded_message}"
    response = requests.get(text_url)
    print(f"Text message sent: {response.status_code}")

# Main function
def main():
    audio_file = record_audio()
    capture_thermal_image() 
    image_path = "/media/pi/CIRCUITPY/images/image.bmp"

    audio_folder = "/home/pi/Developer/reassurance_audio"

    predicted_emotions = process_audio(audio_file)
    thermal_image_bottom_warmer = process_thermal_image(image_path)

    if "fear" in predicted_emotions or thermal_image_bottom_warmer:
        send_telegram_alert()
        play_reassurance_audios(audio_folder)
        send_text_message()
    else:
        print("Safe")

if __name__ == "__main__":
    main()
