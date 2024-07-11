from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .base import Similarity

class SentenceTransformerCosine(Similarity):
    def __init__(self, model: SentenceTransformer, use_cache: bool = True):
        self.model = model
        self.use_cache = use_cache
        self.cache: Dict[str, np.ndarray] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        if self.use_cache and text in self.cache:
            return self.cache[text]

        embedding = self.model.encode(text, convert_to_tensor=True)
        embedding = embedding.cpu().numpy().reshape(1, -1)

        if self.use_cache:
            self.cache[text] = embedding

        return embedding

    def compute(self, text1: str, text2: str) -> float:
        embedding1 = self._get_embedding(text1)
        embedding2 = self._get_embedding(text2)
        return cosine_similarity(embedding1, embedding2)[0][0]

    def clear_cache(self):
        self.cache.clear()