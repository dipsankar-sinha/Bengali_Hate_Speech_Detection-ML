# Define the URI and the new WAV file path
$uri = "http://127.0.0.1:5000/transcribe"
$filePath = "C:\Users\dipsa\Downloads\demo.wav"  # Change to the new WAV file

# Load the required assembly for HttpClient
Add-Type -AssemblyName System.Net.Http

# Create an HttpClient instance
$client = New-Object System.Net.Http.HttpClient

# Create a multipart form data content
$multiContent = New-Object System.Net.Http.MultipartFormDataContent
$fileStream = [System.IO.File]::OpenRead($filePath)
$fileContent = New-Object System.Net.Http.StreamContent($fileStream)
$fileContent.Headers.ContentType = New-Object System.Net.Http.Headers.MediaTypeHeaderValue("audio/wav")  # Update content type for WAV
$multiContent.Add($fileContent, "file", [System.IO.Path]::GetFileName($filePath))

# Send the POST request
$response = $client.PostAsync($uri, $multiContent).Result

# Read and display the response
$responseContent = $response.Content.ReadAsStringAsync().Result
$responseContent