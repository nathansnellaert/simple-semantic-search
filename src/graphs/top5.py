import json
from collections import defaultdict

def load_graph(path, add_values_as_node=False):
    with open(path, "r") as f:
        word_graph = json.load(f)

    # temp fix
    word_graph = {k.lower(): [v.lower() for v in vs] for k, vs in word_graph.items()}
    word_graph = {k: word_graph[k] for k in list(word_graph.keys())}

    reverse_map = defaultdict(list)
    value_counts = defaultdict(int)

    for key, values in word_graph.items():
        for value in values:
            reverse_map[value].append(key)
            value_counts[value] += 1
    
    if add_values_as_node:
        for value, related_words in reverse_map.items():
            if value not in word_graph:
                sorted_related = sorted(related_words, key=lambda x: value_counts[x], reverse=True)
                word_graph[value] = sorted_related[:20]
    
    return word_graph