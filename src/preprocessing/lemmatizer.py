import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import json
from typing import Dict, List

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def nltk_lemmatization(text):
    """
    Lemmatize words using NLTK's WordNetLemmatizer.
    """
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return ' '.join(lemmatized_words)


class CustomGraphLemmatizer:
    def __init__(self, graph_path: str):
        self.lemma_graph = self.load_graph(graph_path)
        self.reverse_lookup = self.create_reverse_lookup()

    def load_graph(self, graph_path: str) -> Dict[str, List[str]]:
        with open(graph_path, 'r') as f:
            return json.load(f)

    def create_reverse_lookup(self) -> Dict[str, str]:
        reverse_lookup = {}
        for lemma, variants in self.lemma_graph.items():
            for variant in variants:
                reverse_lookup[variant] = lemma
            reverse_lookup[lemma] = lemma  # lemma maps to itself
        return reverse_lookup

    def lemmatize(self, word: str) -> str:
        return self.reverse_lookup.get(word.lower(), word.lower())

def custom_graph_lemmatization(text: str, lemmatizer: CustomGraphLemmatizer) -> str:
    """
    Lemmatize words using a custom graph-based lemmatizer.
    """
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)
