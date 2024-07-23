from typing import List, Union
from statistics import mean, median
from .base import Similarity

class EnsembleSimilarity(Similarity):
    def __init__(self, similarity_classes: List[Similarity], aggregation: str = 'mean'):
        """
        Initialize the EnsembleSimilarity class.
        
        :param similarity_classes: List of Similarity subclass instances
        :param aggregation: 'mean' or 'median', determines how to aggregate results
        """
        self.similarity_classes = similarity_classes
        if aggregation not in ['mean', 'median']:
            raise ValueError("aggregation must be either 'mean' or 'median'")
        self.aggregation = aggregation

    def compute(self, phrase1: str, phrase2: str) -> float:
        """
        Compute the similarity between two phrases using all similarity classes
        and return the aggregated result.
        
        :param phrase1: First phrase to compare
        :param phrase2: Second phrase to compare
        :return: Aggregated similarity score
        """
        similarity_scores = [
            sim_class.compute(phrase1, phrase2)
            for sim_class in self.similarity_classes
        ]
        
        if self.aggregation == 'mean':
            return mean(similarity_scores)
        else:  # median
            return median(similarity_scores)

    def add_similarity_class(self, similarity_class: Similarity):
        """
        Add a new similarity class to the ensemble.
        
        :param similarity_class: Similarity subclass instance to add
        """
        self.similarity_classes.append(similarity_class)

    def remove_similarity_class(self, index: int):
        """
        Remove a similarity class from the ensemble by index.
        
        :param index: Index of the similarity class to remove
        """
        if 0 <= index < len(self.similarity_classes):
            del self.similarity_classes[index]
        else:
            raise IndexError("Invalid index for removing similarity class")

    def get_similarity_classes(self) -> List[Similarity]:
        """
        Get the list of current similarity classes in the ensemble.
        
        :return: List of Similarity subclass instances
        """
        return self.similarity_classes

    def set_aggregation_method(self, aggregation: str):
        """
        Set the aggregation method.
        
        :param aggregation: 'mean' or 'median'
        """
        if aggregation not in ['mean', 'median']:
            raise ValueError("aggregation must be either 'mean' or 'median'")
        self.aggregation = aggregation