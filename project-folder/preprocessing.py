# preprocessing.py

import re

def custom_preprocessor(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Remove non-alphabetical characters
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with single space
    return text
