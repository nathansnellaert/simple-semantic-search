import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

def normalize_and_clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove possessive 's
    text = re.sub(r"'s\b", "", text)
    # Replace hyphens with spaces
    text = text.replace('-', ' ')
    # Remove punctuation (except apostrophes within words)
    text = re.sub(r"[^\w\s']", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove apostrophes (after handling possessives)
    text = text.replace("'", "")
    # Strip leading and trailing whitespace
    text = text.strip()
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS]
    text = ' '.join(words)
    return text