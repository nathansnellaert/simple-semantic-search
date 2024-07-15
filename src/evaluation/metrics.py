import random

random.seed(42)

def get_reciprocal_rank(positive_similarities, negative_similarities):
    # Combine all similarities with their corresponding items
    all_items = list(zip(positive_similarities + negative_similarities, 
                         ['positive'] * len(positive_similarities) + 
                         ['negative'] * len(negative_similarities)))

    
    # Return the mean rank if all similarities are the same
    if len(set(sim for sim, _ in all_items)) == 1:
        mean_rank = (len(all_items) + 1) / 2
        return 1 / mean_rank
    
    # Sort by similarity (descending)
    sorted_items = sorted(all_items, key=lambda x: x[0], reverse=True)

    positive_ranks = []
    current_sim = None
    tied_start = 0
    
    for i, (sim, item_type) in enumerate(sorted_items):
        if sim != current_sim:
            if positive_ranks:
                avg_rank = (tied_start + i) / 2
                return 1 / avg_rank
            current_sim = sim
            tied_start = i + 1
        
        if item_type == 'positive':
            positive_ranks.append(i + 1)
    
    if positive_ranks:
        avg_rank = (tied_start + len(sorted_items)) / 2
        return 1 / avg_rank
    
    return 0