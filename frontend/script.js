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
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ "transcription" : textInput })
        });

        let data;
        try {
            data = await response.json();
        } catch (jsonError) {
            throw new Error("Invalid JSON response from the server");
        }

        if (response.ok) {
            let result = data.Hate;
            if (result == 0) {
                result = "It is not a Hate Speech";
            } else {
                result = "It is a Hate Speech";
            }
            resultDiv.innerHTML = `<p><strong>Analysis Result:</strong> ${result}</p>`;
        } else {
            resultDiv.innerHTML = `<p class='error'>Error: ${data.error}</p>`;
        }
    } catch (error) {
        console.error("Error:", error);
        resultDiv.innerHTML = `<p class='error'>Error: ${error.message}</p>`;
    }
}

async function transcribeAudio() {
    const audioInput = document.getElementById("audioInput").files[0];
    if (!audioInput) {
        resultDiv.innerHTML = "<p class='error'>Please upload a WAV file.</p>";
        return;
    }

    // Verify the file type is WAV
    if (audioInput.type !== "audio/wav") {
        resultDiv.innerHTML = "<p class='error'>Only WAV files are supported.</p>";
        return;
    }

    const formData = new FormData();
    formData.append("file", audioInput);

    try {
        const response = await fetch("http://127.0.0.1:5000/transcribe", {
            method: "POST",
            body: formData
        });

        let data;
        try {
            data = await response.json();
        } catch (jsonError) {
            throw new Error("Invalid JSON response from the server");
        }

        if (response.ok) {
            resultDiv.innerHTML = `<p><strong>Transcription:</strong> ${data.transcription}</p>`;
        } else {
            resultDiv.innerHTML = `<p class='error'>Error: ${data.error}</p>`;
        }
    } catch (error) {
        console.error("Error:", error);
        resultDiv.innerHTML = `<p class='error'>Error: ${error.message}</p>`;
    }
}

