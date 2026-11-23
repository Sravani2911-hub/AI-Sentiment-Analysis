#To save and load our trained machine learning model.
import joblib
import pandas as pd #To work with datasets.helps us read, organize, filter, and process data.

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import GridSearchCV #To find suitable model settings automatically.

from preprocess_text import preprocess_text

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Load the processed dataset
df = pd.read_csv("dataset/processed_sentiment_dataset.csv")

# Step 2: Remove missing values
df = df.dropna(subset=["text", "sentiment"])

# Step 3: Clean the text
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Step 3.1: Check duplicate sentences before removing them

print("\nChecking Duplicate Sentences After Preprocessing:")

duplicate_rows = df[
    df.duplicated(subset=["cleaned_text"], keep=False)
]

print("Total duplicate rows:", len(duplicate_rows))

# Check conflicting sentiment labels
conflicts = (
    duplicate_rows.groupby("cleaned_text")["sentiment"]
    .nunique()
)

conflicts = conflicts[conflicts > 1]

print("Sentences with conflicting sentiment labels:", len(conflicts))

# Display conflicting sentences if any exist
if len(conflicts) > 0:
    print("\nConflicting Sentences:")
    print(conflicts)

# Step 3.2: Remove duplicate sentences
df = df.drop_duplicates(subset=["cleaned_text"])

# Step 4: Separate input text and output labels
X = df["cleaned_text"]
y = df["sentiment"]

print("Total Dataset Examples:", len(df))
print("\nSentiment Distribution:")
print(y.value_counts())

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
#step 5.1
target_sentence = "the service made the whole process simple"

print("\n--- TARGET SENTENCE VERIFICATION ---")

print("Training occurrences:",
      (X_train == target_sentence).sum())

print("Testing occurrences:",
      (X_test == target_sentence).sum())

print("\nTraining labels:")
print(y_train[X_train == target_sentence])

print("\nTesting labels:")
print(y_test[X_test == target_sentence])


#step 5.2
target_sentence = "the service made the whole process simple"

print("\nTarget Sentence Details:")

matching_rows = df[df["cleaned_text"] == target_sentence]

print(matching_rows[["text", "cleaned_text", "sentiment"]])

print("\nTraining label:")
print(y_train[X_train == target_sentence])

#step 5.3
target_sentence = "the service made the whole process simple"

print("\nChecking Target Sentence:")

if target_sentence in X_train.values:
    print("Target sentence is in TRAINING data")
elif target_sentence in X_test.values:
    print("Target sentence is in TESTING data")
else:
    print("Target sentence was NOT found")


#step 5.4
# Inspect word distribution in training and testing data

training_data = pd.DataFrame({
    "text": X_train,
    "sentiment": y_train
})

testing_data = pd.DataFrame({
    "text": X_test,
    "sentiment": y_test
})

words_to_check = [
    "tasty",
    "ordinary",
    "upset",
    "promised",
    "noisy",
    "unhappy",
    "disappointing",
    "helpful",
    "brilliantly",
    "uncomfortable"
]

for word in words_to_check:

    print(f"\nWord: {word}")

    for name, data in [
        ("Training Data", training_data),
        ("Testing Data", testing_data)
    ]:

        matching = data[
            data["text"].str.contains(
                word,
                case=False,
                regex=False,
                na=False
            )
        ]

        print(f"\n{name}: {len(matching)} occurrences")

        if len(matching) > 0:
            print(matching["sentiment"].value_counts().to_string())


target_sentence = "i love my village because the people are kind"

# Check whether the sentence is in the training data
train_mask = X_train == target_sentence

if train_mask.any():
    print("Training label:", y_train[train_mask].values)
else:
    print("Target sentence is not in the training set.")

# Check whether it is in the testing data
test_mask = X_test == target_sentence

if test_mask.any():
    print("Testing label:", y_test[test_mask].values)
else:
    print("Target sentence is not in the testing set.")

print("\nChecking Target Sentence:")
print("In Training Set:", target_sentence in X_train.values)
print("In Testing Set:", target_sentence in X_test.values)


check_sentences = [
    "the service is not good at all",
    "nothing unusual happened during the day"
]

print("\nChecking New Training Examples:")

for sentence in check_sentences:
    print("\nSentence:", sentence)
    print("In Training Set:", sentence in X_train.values)
    print("In Testing Set:", sentence in X_test.values)
    

print("\nTraning Examples:", len(X_train))
print("Testing Examples:", len(X_test))

# Step 6: Build a Pipeline for TF-IDF and Logistic Regression

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Step 7: Tune TF-IDF and Logistic Regression together
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2],
    "classifier__C": [0.001, 0.01, 0.1, 1],
    "classifier__class_weight": [None, "balanced"]
}

#GridSearchCV: Tests different parameter combinations and selects the best-performing configuration.
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="f1_macro"
)

grid_search.fit(X_train, y_train)

#..
# Compare the best CV score for each ngram range
import pandas as pd

results = pd.DataFrame(grid_search.cv_results_)

ngram_scores = results.groupby(
    "param_tfidf__ngram_range"
)["mean_test_score"].max()

print("\nComparing Ngram Performance:")

for ngram, score in ngram_scores.items():
    print(f"Ngram Range: {ngram} | CV F1 Score: {score * 100:.2f}%")



# Step 8: Select the best model
model = grid_search.best_estimator_

# Check whether important positive words exist in TF-IDF vocabulary
vocabulary = model.named_steps["tfidf"].vocabulary_

important_words = [
    "love",
    "kind",
    "village",
    "beautiful",
    "excellent",
    "satisfied",
    "enjoyed",
    "below",
    "expectations",
    "expected",
    "hotel"
]

print("\nChecking Words in Training Data:")

training_data = pd.DataFrame({
    "text": X_train,
    "sentiment": y_train
})

for word in important_words:

    found_examples = training_data[
        training_data["text"].str.contains(
            r"\b" + word + r"\b",
            regex=True
        )
    ]

    print(f"\nWord: {word}")
    print("Number of sentences:", len(found_examples))

    for _, row in found_examples.iterrows():
        print(f"- [{row['sentiment']}] {row['text']}")

print("\nChecking Positive Words in Vocabulary:")

for word in important_words:
    if word in vocabulary:
        print(word, "-> Found")
    else:
        print(word, "-> Not Found")


# Calculate training accuracy
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)

print("Training Accuracy:", round(train_accuracy * 100, 2), "%")
print("\nBest Model Parameters:", grid_search.best_params_)
print(
    "Best Cross-Validation F1 Score:",
    round(grid_search.best_score_ * 100, 2),
    "%"
)

print("\nModel training completed successfully!")

# Step 9: Predict sentiments for the testing data
y_pred = model.predict(X_test)

# Step 9.1: Display incorrect predictions
print("\nIncorrect Predictions:")
print("-" * 70)

for text, actual, predicted in zip(X_test, y_test, y_pred):
    if actual != predicted:
        print("Sentence:", text)
        print("Actual Sentiment:", actual)
        print("Predicted Sentiment:", predicted)
        print("-" * 70)

# Step 10: Calculate accuracy - Accuracy = (5 / 6) × 100 = 83.33%
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")


# Step 11: Display the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 11.1: Display the confusion matrix
# Generate confusion matrix
cm = confusion_matrix(
    y_test,
    grid_search.predict(X_test),
    labels=["negative", "neutral", "positive"]
)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"]
)

display.plot()
plt.title("Sentiment Analysis Confusion Matrix")
plt.show()

#step 11.2
# Display incorrectly predicted sentences

predictions = grid_search.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions,
    "Sentence": X_test
})

incorrect = results[
    results["Actual"] != results["Predicted"]
]

print("\nIncorrectly Predicted Sentences:")
print(incorrect.to_string(index=False))


# Step 12: Test new sentences

test_sentences = [
    "I love this.",
    "The people are kind.",
    "I love my village.",
    "My village is beautiful.",
    "I love my village because the people are kind.",
    "The service was excellent.",
    "I am satisfied with the product.",
    "I really enjoyed the experience."
]

print("\nTesting New Sentences:")
print("-" * 50)

for sentence in test_sentences:
    cleaned_sentence = preprocess_text(sentence)

    prediction = model.predict([cleaned_sentence])
    probabilities = model.predict_proba([cleaned_sentence])[0]

    class_probabilities = dict(zip(model.classes_, probabilities))

    print("\nSentence:", sentence)
    print("Predicted Sentiment:", prediction[0])
    print("Sentiment Scores:")

    for sentiment, score in class_probabilities.items():
        print(f"  {sentiment}: {score * 100:.2f}%")

print("\n" + "-" * 50)

# Step 13: Save the trained model
joblib.dump(model, "sentiment_model.pkl")

print("\nModel saved successfully!")

