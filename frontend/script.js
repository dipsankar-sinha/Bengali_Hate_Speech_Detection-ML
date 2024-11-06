const resultDiv = document.getElementById("result");

async function processText() {
    const textInput = document.getElementById("textInput").value;
    if (!textInput) {
        resultDiv.innerHTML = "<p class='error'>Please enter some text to analyze.</p>";
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:5000/process_input", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ text: textInput })
        });
        const data = await response.json();
        if (response.ok) {
            resultDiv.innerHTML = <p><strong>Analysis Result:</strong> ${JSON.stringify(data.predictions)}</p>;
        } else {
            resultDiv.innerHTML = <p class='error'>Error: ${data.error}</p>;
        }
    } catch (error) {
        resultDiv.innerHTML = <p class='error'>Error: ${error.message}</p>;
    }
}

async function transcribeAudio() {
    const audioInput = document.getElementById("audioInput").files[0];
    if (!audioInput) {
        resultDiv.innerHTML = "<p class='error'>Please upload a WAV file.</p>";
        return;
    }

    const formData = new FormData();
    formData.append("file", audioInput);

    try {
        const response = await fetch("http://127.0.0.1:5000/transcribe", {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            resultDiv.innerHTML = <p><strong>Transcription:</strong> ${data.transcription}</p>;
        } else {
            resultDiv.innerHTML = <p class='error'>Error: ${data.error}</p>;
        }
    } catch (error) {
        resultDiv.innerHTML = <p class='error'>Error: ${error.message}</p>;
    }
}
