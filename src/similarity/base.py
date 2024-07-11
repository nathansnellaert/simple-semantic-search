from abc import ABC, abstractmethod

class Similarity(ABC):
    @abstractmethod
    def compute(self, text1: str, text2: str) -> float:
        """
        Compute the similarity between two texts.
        
        :param text1: The first text to compare
        :param text2: The second text to compare
        :return: A float representing the similarity score (typically between 0 and 1)
        """
        pass