import pandas as pd
from preprocess_text import preprocess_text

# Step 1: Load the expanded dataset
df = pd.read_csv("dataset/expanded_sentiment_dataset.csv")

# Remove extra spaces from sentiment labels
df["sentiment"] = df["sentiment"].str.strip().str.lower()

print("\nUnique Sentiment Labels:")
print(df["sentiment"].unique())

# Step 2: Check the dataset columns
print("Original Dataset Shape:", df.shape)
print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

# Step 3: Remove duplicate sentences
df = df.drop_duplicates(subset=["text"])

# Step 4: Remove rows with missing values
df = df.dropna(subset=["text", "sentiment"])

# Step 5: Clean the text
df["cleaned_text"] = df["text"].apply(preprocess_text)

# Step 6: Save the processed dataset
df.to_csv("dataset/processed_sentiment_dataset.csv", index=False)

# Step 7: Display the results
print("\nProcessed Dataset Shape:", df.shape)
print("\nProcessed Dataset Preview:")
print(df[["text", "cleaned_text", "sentiment"]].head())

print("\nPreprocessing completed successfully!") 