from .base import Similarity
from ..preprocessing.preprocess import normalize_and_clean_text


class JaccardSimilarity(Similarity):
    def __init__(self):
        pass

    def compute(self, phrase1: str, phrase2: str) -> float:
        # Normalize and split the phrases into words
        words1 = set(normalize_and_clean_text(phrase1))
        words2 = set(normalize_and_clean_text(phrase2))

        # Find the intersection (common words)
        common_words = words1.intersection(words2)

        # Calculate similarity as the ratio of common words to total unique words
        total_unique_words = len(words1.union(words2))
        
        if total_unique_words == 0:
            return 0.0 
        
        similarity = len(common_words) / total_unique_words
        return similarity