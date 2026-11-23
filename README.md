# AI Sentiment Analysis System

## 1. Project Overview

The AI Sentiment Analysis System is a machine learning-based web application that analyzes text and predicts its sentiment.

It classifies text into three categories:

- Positive
- Negative
- Neutral

The application uses Natural Language Processing (NLP) and Machine Learning to analyze text and display the predicted sentiment along with probability scores.

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

- Predicts positive, negative, and neutral sentiments.
- Displays probability scores for all three sentiment categories.
- Preprocesses text before prediction.
- Provides a live character counter with a 1,000-character limit.
- Includes ready-made example sentences.
- Allows users to copy sentiment analysis results.
- Provides a Clear button to reset the input and results.
- Displays sentiment results using different colors.
- Connects the frontend and backend using FastAPI.

## 4. Project Structure

```text
AI-Sentiment-Analysis/
│
├── dataset/
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
├── sentiment_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## 5. How the System Works

1. The dataset is prepared and cleaned using text preprocessing.
2. TF-IDF converts text into numerical features.
3. Logistic Regression learns sentiment patterns from the training data.
4. The trained model is saved as a `.pkl` file.
5. FastAPI loads the trained machine learning model.
6. The user enters text through the frontend.
7. JavaScript sends the text to the FastAPI backend.
8. The backend preprocesses the input and predicts its sentiment.
9. The predicted sentiment and probability scores are returned to the frontend.

## 6. How to Run the Project

### Step 1: Create and Activate the Virtual Environment

```cmd
python -m venv .venv
.venv\Scripts\activate.bat


### Step 2: Install Required Packages

```cmd
pip install -r requirements.txt


### Step 3: Start the Backend

```cmd
python -m uvicorn main:app --reload


Backend URL:

http://127.0.0.1:8000

API Documentation:

http://127.0.0.1:8000/docs

### Step 4: Run the Frontend

1. Open the project folder in VS Code.
2. Open the `frontend` folder.
3. Right-click `index.html`.
4. Select **Open with Live Server**.

The frontend will open in your browser.

## 7. Machine Learning Model

The project uses:

- **TF-IDF Vectorization:** Converts text into numerical features.
- **Logistic Regression:** Classifies text into sentiment categories.
- **Joblib:** Saves and loads the trained machine learning model.

The model predicts one of three sentiment categories: Positive, Negative, or Neutral.

## 8. Project Objective

The objective of this project is to understand and implement a complete machine learning application, from dataset preprocessing and model training to API development and frontend integration. 