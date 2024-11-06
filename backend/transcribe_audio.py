import requests

# Define the URI and the WAV file path
uri = "http://127.0.0.1:5000/transcribe"
file_path = r"C:\Users\dipsa\Downloads\demo.wav"  # Change to the new WAV file

# Open the WAV file in binary mode and prepare the multipart form data
with open(file_path, 'rb') as file:
    files = {'file': (file_path.split('\\')[-1], file, 'audio/wav')}
    
    # Send the POST request
    response = requests.post(uri, files=files)

# Read and display the response
print(response.json())
