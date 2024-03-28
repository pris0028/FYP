import cv2
import numpy as np
import os

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
    elif top_mean_temp < bottom_mean_temp:
        print("Bottom half is warmer")
    else:
        print("Both halves have similar temperatures")

if __name__ == "__main__":
    folder_path = "/media/pi/CIRCUITPY/MLX90640/"
    file_count = sum(1 for file in os.listdir(folder_path) if file.endswith(".bmp") and not file.startswith("."))
    
    for i in range(1, file_count + 1):
        image_path = f"{folder_path}frm{str(i).zfill(5)}.bmp"
        compare_temperature(image_path)
