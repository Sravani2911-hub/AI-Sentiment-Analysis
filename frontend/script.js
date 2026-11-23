const textInput = document.getElementById("textInput");
const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");
const result = document.getElementById("result");
const characterCount = document.getElementById("characterCount");
const exampleButtons = document.querySelectorAll(".example-button");
const copyButton = document.getElementById("copyButton");
const resultContainer = document.getElementById("resultContainer");

// Get the sentiment score elements
const scores = document.getElementById("scores");
const positiveScore = document.getElementById("positiveScore");
const negativeScore = document.getElementById("negativeScore");
const neutralScore = document.getElementById("neutralScore");

// Update character count while typing
textInput.addEventListener("input", function () {
    characterCount.textContent =
        textInput.value.length + " / 1000 characters";
});

// Analyze sentiment when the button is clicked
analyzeButton.addEventListener("click", async function () {
    const text = textInput.value.trim();

    if (text === "") {
        result.textContent = "Please enter some text to analyze.";
        result.className = "result";
        resultContainer.className = "result-container";
        scores.style.display = "none";
        return;
    }

    result.textContent = "Analyzing sentiment, please wait...";
    result.className = "result";
    resultContainer.className = "result-container";
    scores.style.display = "none";

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (!response.ok) {
            const message = data.detail || "Unable to analyze the text.";

            throw new Error(
                typeof message === "string"
                    ? message
                    : "Please enter valid text and try again."
            );
        }

        if (data.error) {
            result.textContent = data.error;
            return;
        }

        // Display the predicted sentiment
        const sentiment = data.sentiment.toLowerCase();

        result.textContent =
            "Predicted Sentiment: " + sentiment.toUpperCase();

       resultContainer.className = "result-container " + sentiment;

        // Display sentiment probability scores
        positiveScore.textContent =
            data.sentiment_scores.positive + "%";

        negativeScore.textContent =
            data.sentiment_scores.negative + "%";

        neutralScore.textContent =
            data.sentiment_scores.neutral + "%";

        scores.style.display = "block";
        copyButton.style.display = "block";

    } catch (error) {
        result.textContent =
            error.message || "Unable to connect to the sentiment analysis API.";

        console.error("Sentiment analysis error:", error);

    } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Sentiment";
    }
});

// Clear the input text and previous analysis result
clearButton.addEventListener("click", function () {
    textInput.value = "";

    characterCount.textContent = "0 / 1000 characters"; 

    result.textContent = "Please enter some text to analyze.";
    result.className = "result";
    resultContainer.className = "result-container";

    scores.style.display = "none";
    copyButton.style.display = "none";

    positiveScore.textContent = "0%";
    negativeScore.textContent = "0%";
    neutralScore.textContent = "0%";

    textInput.focus();
});

// Fill the input box when an example button is clicked

exampleButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        textInput.value = button.dataset.text;

        // Update the character counter
        characterCount.textContent =
            textInput.value.length + " / 1000 characters";

        textInput.focus();
    });
});


// Copy sentiment analysis result
copyButton.addEventListener("click", async function () {
    const copiedText =
        result.textContent + "\n\n" +
        "Positive: " + positiveScore.textContent + "\n" +
        "Negative: " + negativeScore.textContent + "\n" +
        "Neutral: " + neutralScore.textContent;

    try {
        await navigator.clipboard.writeText(copiedText);

        copyButton.textContent = "Copied!";

        setTimeout(function () {
            copyButton.textContent = "Copy Result";
        }, 2000);

    } catch (error) {
        console.error("Unable to copy result:", error);
        copyButton.textContent = "Copy Failed";

        setTimeout(function () {
            copyButton.textContent = "Copy Result";
        }, 2000);
    }
});