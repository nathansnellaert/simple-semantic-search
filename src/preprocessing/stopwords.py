import nltk
from nltk.corpus import stopwords

# Download the NLTK stopwords data
nltk.download('stopwords', quiet=True)

def nltk_stopword_removal(text):
    """
    Remove stopwords using NLTK's English stopwords list.
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)