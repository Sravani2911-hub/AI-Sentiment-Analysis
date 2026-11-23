
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib

from text_preprocessing import preprocess_text

# Step 1: Create the FastAPI application
app = FastAPI(
    title="AI Sentiment Analysis API",
    description="An API that predicts the sentiment of text",
    version="1.0.0"
)

# Step 2: Configure CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Step 3: Load the complete trained Pipeline
model = joblib.load("sentiment_model.pkl")


# Step 4: Define the incoming request structure
class SentimentRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=1000,
        description="Enter text between 1 and 1000 characters"
    )


# Step 5: Create the home endpoint
@app.get("/")
def home():
    return {
        "message": "AI Sentiment Analysis API is running!"
    }


# Step 6: Create the sentiment prediction endpoint
@app.post("/predict")
def predict_sentiment(request: SentimentRequest):

    # Get the original text
    original_text = request.text

    # Reject empty or whitespace-only text
    if not original_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter meaningful text to analyze."
        )

    # Clean the text
    cleaned_text = preprocess_text(original_text)

    # Check whether preprocessing removed all meaningful text
    if not cleaned_text.strip():
        raise HTTPException(
            status_code=400,
            detail="No meaningful text remains after preprocessing."
        )

    # Predict sentiment using the complete Pipeline
    prediction = model.predict([cleaned_text])[0]

    # Get the model's probability estimates
    probabilities = model.predict_proba([cleaned_text])[0]

    # Match each sentiment with its probability
    sentiment_scores = {
        sentiment: round(float(score) * 100, 2)
        for sentiment, score in zip(model.classes_, probabilities)
    }

    # Return the prediction and scores
    return {
        "input_text": original_text,
        "cleaned_text": cleaned_text,
        "sentiment": str(prediction),
        "sentiment_scores": sentiment_scores
    }

