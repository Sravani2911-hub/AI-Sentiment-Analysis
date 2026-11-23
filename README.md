# AI Sentiment Analysis System

## 1. Project Overview

The AI Sentiment Analysis System is a machine learning-based web application that analyzes text and predicts its sentiment.

It classifies text into three categories:

- Positive
- Negative
- Neutral

The application uses Natural Language Processing (NLP) and Machine Learning to understand text and display the predicted sentiment along with probability scores.

## 2. Technologies Used

**Backend:**
- Python
- FastAPI

**Machine Learning and NLP:**
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

**Frontend:**
- HTML
- CSS
- JavaScript

**Other Tools:**
- Joblib
- VS Code
- Git and GitHub

## 3. Features

- Analyzes user-entered text.
- Predicts positive, negative, or neutral sentiment.
- Displays sentiment probability scores.
- Preprocesses text before prediction.
- Uses a trained machine learning model.
- Provides a simple web interface.
- Connects the frontend with the backend using a FastAPI API.

## 4. Project Structure

```text
AI-Sentiment-Analysis/
│
├── dataset/
│   ├── sentiment_dataset.csv
│   ├── expanded_sentiment_dataset.csv
│   └── processed_sentiment_dataset.csv
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── main.py
├── preprocess_dataset.py
├── text_preprocessing.py
├── train_model.py
├── inspect_prediction.py
├── explore_dataset.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── README.md
```

## 5. How the System Works

1. The dataset is prepared and cleaned using text preprocessing.
2. TF-IDF converts text into numerical features.
3. Logistic Regression learns sentiment patterns from the training data.
4. The trained model is saved as a `.pkl` file.
5. FastAPI loads the saved machine learning model.
6. The user enters text through the frontend.
7. JavaScript sends the text to the FastAPI backend.
8. The backend preprocesses the input and predicts its sentiment.
9. The predicted sentiment and probability scores are returned to the frontend.

## 6. How to Run the Project

### Step 1: Activate the Virtual Environment

```cmd
.venv\Scripts\activate.bat


### Step 2: Install Required Packages

```cmd
pip install fastapi uvicorn pandas scikit-learn joblib


### Step 3: Start the Backend

```cmd
python -m uvicorn main:app --reload


Backend URL:

http://127.0.0.1:8000

API Documentation:

http://127.0.0.1:8000/docs

### Step 4: Run the Frontend

Open the `frontend` folder in VS Code.

Right-click `index.html` and select **Open with Live Server**.

## 7. Machine Learning Model

The project uses:

- TF-IDF Vectorization to convert text into numerical features.
- Logistic Regression to classify the sentiment.
- A trained model saved using Joblib.

The model predicts one of three sentiment categories: Positive, Negative, or Neutral.

## 8. Project Objective

The objective of this project is to understand and implement a complete machine learning application, from dataset preprocessing and model training to API development and frontend integration.