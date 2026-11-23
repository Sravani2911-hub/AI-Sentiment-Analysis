# import re

# def preprocess_text(text):
#     # Convert all letters to lowercase.
#     text = text.lower()

#     # Remove URLs.
#     text = re.sub(r"https?://\S+|www\.\S+", "", text)

#     # Remove punctuation and special characters.
#     text = re.sub(r"[^\w\s]", "", text)

#     # Remove extra spaces.
#     text = " ".join(text.split())

#     return text

import re

def preprocess_text(text):

    # Step 1: Convert text to lowercase
    text = text.lower()

    # Step 2: Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Step 3: Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", "", text)

    # Step 4: Remove extra spaces
    text = " ".join(text.split())

    return text


# Test negative words and contractions
test_sentences = [
    "This is not good!",
    "I never liked this product.",
    "I don't like this movie.",
    "I didn't enjoy the service.",
    "This isn't working properly."
]

for sentence in test_sentences:
    cleaned_text = preprocess_text(sentence)

    print("Original:", sentence)
    print("Cleaned:", cleaned_text)
    print("--------------------------")