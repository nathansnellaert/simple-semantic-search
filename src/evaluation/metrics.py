import random

random.seed(42)

def get_reciprocal_rank(positive_similarities, negative_similarities):
    # Combine all similarities with their corresponding items
    all_items = list(zip(positive_similarities + negative_similarities, 
                         ['positive'] * len(positive_similarities) + 
                         ['negative'] * len(negative_similarities)))
    
    # Sort by similarity (descending) and shuffle to break ties
    random.shuffle(all_items)
    sorted_items = sorted(all_items, key=lambda x: x[0], reverse=True)
    
    # Check if all similarities are the same
    if len(set(sim for sim, _ in sorted_items)) == 1:
        return None
    
    # Find the highest rank (lowest index) of any positive item
    ranks = [i + 1 for i, (sim, item_type) in enumerate(sorted_items) if item_type == 'positive']
    best_rank = min(ranks) if ranks else 0
    
    return 1 / best_rank if best_rank > 0 else 0