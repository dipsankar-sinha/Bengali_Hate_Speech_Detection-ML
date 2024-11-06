from flask import Flask, request, jsonify
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor

import requests
import torch
import librosa
import pickle

app = Flask(__name__)

# Load the model (Logistic Regressiomn, Random Forest, SVM Model, Gradient Boosting)
with open("model/hs_models/Logistic Regression_model.pkl", "rb") as f:
    lr_model = pickle.load(f)
with open("model/hs_models/Random Forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("model/hs_models/SVM_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("model/hs_models/Gradient Boosting_model.pkl", "rb") as f:
    gb_model = pickle.load(f)

# Load the TF-IDF Vectorizer
with open("model/hs_models/vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load the processor
with open("model/asr/BanglaASR_processor.pkl", "rb") as f:
    processor = pickle.load(f)

# The Model Path
model_path = "bangla-speech-processing/BanglaASR"

# Load the model
speech_model = WhisperForConditionalGeneration.from_pretrained("bangla-speech-processing/BanglaASR")
speech_model.load_state_dict(torch.load("model/asr/BanglaASR_state.pth"))
speech_model.eval()  # Important for inference

# Load the Feature Extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

@app.route('/transcribe', methods=['GET','POST'])
def transcribe_audio():
    if request.method == "POST":
        try:
            # Get the uploaded audio file
            file = request.files['file']
            
            # Load the audio file using librosa
            audio_input, _ = librosa.load(file, sr=16000)  # Ensure the sampling rate is consistent

            # Preprocess audio to get input features
            input_features = feature_extractor(audio_input, return_tensors="pt", sampling_rate=16000).input_features
            current_length = input_features.shape[-1]
            target_length = 3000

            # Pad the mel spectrogram if necessary
            if current_length < target_length:
                pad_length = target_length - current_length
                input_features = torch.nn.functional.pad(input_features, (0, pad_length), mode="constant", value=0)

            # Move features and model to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_features = input_features.to(device)
            speech_model.to(device)

            # Get model predictions
            with torch.no_grad():
                predicted_ids = speech_model.generate(inputs=input_features, language="bn")

            # Decode predictions
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            return jsonify({"transcription": transcription})

        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return '<h1 style="text-align = center">API for Bengali ASR<h1>'
    
@app.route('/detect', methods=['GET','POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    file.save('uploaded_recording.wav')  # Save the file to disk, or process it directly

    # Add further processing here (e.g., transcribing, analysis, etc.)
    return jsonify({'message': 'File received successfully'})
    
     # Call your own API using `requests`
    
    
    # Check the response status and return the data
    if response.status_code == 200:
        api_data = response.json()
        return jsonify({
            "message": "Successfully called the API!",
            "api_response": api_data
        })
    else:
        return jsonify({"error": "Failed to call the API"}), 500
    return

@app.route('/process_input', methods=['POST'])
def process_input():

    #Hate Speech Detect the URI
    detect_uri = "http://127.0.0.1:5000/transcribe"

    # Check if the request contains a text field
    if 'text' in request.form:
        text = request.form['text']
        
        response_detect = requests.post()
        return jsonify({'message': 'Text received successfully', 'text': text})

    # Check if the request contains an audio file
    elif 'file' in request.files:
        file = request.files['file']

        #file.save('uploaded_audio.wav')  # Save the audio file, or process it directly

        transcribe_uri = "http://127.0.0.1:5000/transcribe"
    
        # Check if the file is in WAV format
        if file.content_type != 'audio/wav':
            return jsonify({'error': 'Only WAV format is accepted'}), 400
        
        response_transcribe = requests.post(transcribe_uri, files=file)
        response_detect = requests.post(transcribe_uri, json=response_transcribe.json())
        
        return response_detect.json()

    else:
        return jsonify({'error': 'No valid input found'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
