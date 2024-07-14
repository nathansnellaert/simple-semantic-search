import re

def lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_oov(text, vocabulary):
    return ' '.join([word for word in text.split() if word in vocabulary])