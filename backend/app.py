from flask import Flask, request, jsonify
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor
from nltk.corpus import stopwords
from bangla_stemmer.stemmer.stemmer import BanglaStemmer
import re
import unicodedata
import nltk
import requests
import torch
import librosa
import pickle

app = Flask(__name__)


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

### Data Cleaning and Normalisation of the text for hate speech detection###

# Initialize stopwords and stemmer for Bengali

stop_words = set(stopwords.words('bengali'))
stemmer = BanglaStemmer()

# Regex pattern to keep Bengali characters and whitespace
chars_to_ignore_regex = r'[^\u0980-\u09FF\s]'


def clean_text(text):
    # Normalize Unicode and remove unwanted characters
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(chars_to_ignore_regex, '', text)
    text = re.sub(r'[\u09E6-\u09EF]', '', text)  # Remove Bengali numbers
    text = text.lower()  # Convert to lowercase

    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Apply stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    text = nltk.word_tokenize(text)

    # Finialy converting the bengali text into more simplier form

    y = []
    for i in text:
        i = re.sub(r'\W+', '', i)
        if i:
            y.append(i)

    return " ".join(y)

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

# URIs for transcription and detect API 
TRANSCRIBE_URI = "http://127.0.0.1:5000/transcribe"
DETECT_URI = "http://127.0.0.1:5000/detect"

# Load the model
speech_model = WhisperForConditionalGeneration.from_pretrained("bangla-speech-processing/BanglaASR")
speech_model.load_state_dict(torch.load("model/asr/BanglaASR_state.pth"))
speech_model.eval()  # Important for inference

# Load the Feature Extractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

@app.route('/transcribe', methods=['GET','POST'])
def transcribe_audio():
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
        app.logger.error(e)
        return jsonify({"error": str(e)}), 500
    
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()

        text = data['transcription']
        norm_text = clean_text(text)

        #Feature Extraction using TF-IDF vectorizer 
        feature_text = tfidf_vectorizer.transform([norm_text])
        result = lr_model.predict(feature_text)
        app.logger.info(result)
        return jsonify({"Hate" : str(result[0])})
    except Exception as e:
        app.logger.error(e)
        return jsonify({"error": str(e)}), 500


@app.route('/process_input', methods=['POST'])
def process_input():
    # Check if the request contains text
    if 'text' in request.form:
        text = request.form['text']
        response_detect = requests.post(DETECT_URI, json=jsonify({'transcription': text}))
        return response_detect.json()

    # Check if the request contains an audio file
    elif 'file' in request.files:
        file = request.files['file']

        # Ensure the file is in WAV format
        if file.content_type != 'audio/wav':
            return jsonify({'error': 'Only WAV format is accepted'}), 400

        # Send the audio file for transcription
        response_transcribe = requests.post(TRANSCRIBE_URI, files={'file': file})

        if response_transcribe.ok:
            response_detect = requests.post(DETECT_URI, json=response_transcribe.json())
            return response_detect.json() if response_detect.ok else jsonify({"error": "Failed to call the API"}), 500

        return jsonify({"error": "Transcription API failed"}), 500

    # No valid input found
    return jsonify({'error': 'No valid input found'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
