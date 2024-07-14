from nltk.stem import PorterStemmer

def stem_words(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split()])