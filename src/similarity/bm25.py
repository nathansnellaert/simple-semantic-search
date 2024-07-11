import math
from collections import Counter
from typing import List, Dict
from .base import Similarity

class BM25Similarity(Similarity):
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.corpus = [doc.split() for doc in corpus]
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / self.N
        self.doc_freqs = self._calculate_doc_freqs()
        self.idf = self._calculate_idf()
        self.doc_len = [len(doc) for doc in self.corpus]

    def _calculate_doc_freqs(self) -> Dict[str, int]:
        doc_freqs = Counter()
        for doc in self.corpus:
            doc_freqs.update(set(doc))
        return doc_freqs

    def _calculate_idf(self) -> Dict[str, float]:
        idf = {}
        for word, freq in self.doc_freqs.items():
            idf[word] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def _score(self, query: List[str], doc_id: int) -> float:
        score = 0.0
        doc = self.corpus[doc_id]
        doc_len = self.doc_len[doc_id]
        for word in query:
            if word not in doc:
                continue
            freq = doc.count(word)
            numerator = self.idf[word] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += numerator / denominator
        return score

    def compute(self, query: str, document: str) -> float:
        query_terms = query.split()
        doc_terms = document.split()
        
        # Add the document to the corpus temporarily
        self.corpus.append(doc_terms)
        self.doc_len.append(len(doc_terms))
        self.N += 1
        
        # Update doc_freqs and idf
        for term in set(doc_terms):
            self.doc_freqs[term] += 1
        self.idf = self._calculate_idf()
        
        # Calculate BM25 score
        score = self._score(query_terms, len(self.corpus) - 1)
        
        # Remove the document from the corpus
        self.corpus.pop()
        self.doc_len.pop()
        self.N -= 1
        
        # Revert doc_freqs and idf
        for term in set(doc_terms):
            self.doc_freqs[term] -= 1
            if self.doc_freqs[term] == 0:
                del self.doc_freqs[term]
        self.idf = self._calculate_idf()
        
        return score