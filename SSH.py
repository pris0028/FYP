import requests
from datetime import datetime
import urllib.parse
import time

# Set the username
username = "imprisla"

# Get the current date and time
now = datetime.now()
date_time = now.strftime("%Y-%m-%d %H:%M:%S")

# Construct the message
message = "A fall has taken place at " + date_time

# URL encode the message
encoded_message = urllib.parse.quote(message)

# Send the text message
text_url = f"https://api.callmebot.com/text.php?user=@{username}&text={encoded_message}"
response = requests.get(text_url)
print(f"Text message sent: {response.status_code}")

# Wait for 65 seconds
time.sleep(65)

# Make the voice call
voice_url = f"https://api.callmebot.com/start.php?user=@{username}&text={encoded_message}"
response = requests.get(voice_url)
print(f"Voice call initiated: {response.status_code}")