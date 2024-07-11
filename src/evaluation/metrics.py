import random

import random

def get_reciprocal_rank(positive_similarities, negative_similarities):
    """
    Calculate the Mean Reciprocal Rank (MRR) for a query based on pre-calculated similarities.
    
    This implementation uses a linear scan approach for improved efficiency.
    It treats all matches scored 0 as the lowest possible score, adhering to
    the traditional MRR definition for simplicity and deterministic results.

    Args:
    positive_similarities (list): List of similarity scores for positive items
    negative_similarities (list): List of similarity scores for negative items

    Returns:
    float: The reciprocal rank of the highest-ranked positive item, or 0 if none found
    """
    
    # Combine similarities with their types
    all_items = [(sim, True) for sim in positive_similarities] + \
                [(sim, False) for sim in negative_similarities]
    
    # Track the number of items with higher similarity
    higher_count = 0
    
    # Find the highest similarity score
    max_similarity = max(sim for sim, _ in all_items)
    
    # If max similarity is 0, return 0 (all items have 0 similarity)
    if max_similarity == 0:
        return 0
    
    # Scan through items
    for similarity, is_positive in all_items:
        if similarity == max_similarity:
            if is_positive:
                return 1 / (higher_count + 1)
            higher_count += 1
        elif similarity < max_similarity:
            if is_positive:
                return 1 / (higher_count + 1)
            higher_count += 1
    
    # If no positive item is found, return 0
    return 0