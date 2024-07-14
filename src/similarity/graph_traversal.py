from .base import Similarity
from preprocessing.preprocess import normalize_and_clean_text
from collections import defaultdict

class GraphTraversalSimilarity(Similarity):
    def __init__(self, word_graph, max_depth, damping_factor=0.85):
        self.word_graph = word_graph
        self.max_depth = max_depth
        self.damping_factor = damping_factor

    def compute(self, phrase1: str, phrase2: str) -> float:
        words1 = set((phrase1).split())
        words2 = set((phrase2).split())
        
        direct_matches = words1.intersection(words2)
        direct_match_score = len(direct_matches) / max(len(words1), len(words2))
        
        remaining_words1 = words1 - direct_matches
        remaining_words2 = words2 - direct_matches
        
        traversal1 = self._calculate_graph_traversal(remaining_words1)
        traversal2 = self._calculate_graph_traversal(remaining_words2)
        
        all_words = set(traversal1.keys()) | set(traversal2.keys())
        
        traversal_similarity = 0
        total_weight = 0
        for word in all_words:
            t1 = traversal1.get(word, 0)
            t2 = traversal2.get(word, 0)
            if t1 == 0 or t2 == 0:
                continue
            traversal_similarity += 1
            total_weight += 1
        
        normalized_traversal_similarity = traversal_similarity / total_weight if total_weight > 0 else 0
        
        final_similarity = (direct_match_score + normalized_traversal_similarity) / 2
        
        return final_similarity

    def _calculate_graph_traversal(self, words):
        traversal_scores = defaultdict(float)
        
        for word in words:
            neighbors = self._get_extended_neighbors(self.word_graph, word, self.max_depth)
            
            for neighbor, depth in neighbors.items():
                neighbor_score = (self.damping_factor ** depth)
                traversal_scores[neighbor] += neighbor_score
        
        return traversal_scores
    
    def _get_extended_neighbors(self, word_graph, word, max_depth):
        neighbors = {}
        queue = [(word, 0)]
        visited = set()

        while queue:
            current_word, depth = queue.pop(0)
            if depth > max_depth:
                break
            
            if current_word not in visited:
                visited.add(current_word)
                current_neighbors = word_graph.get(current_word, [])
                for neighbor in current_neighbors:
                    if neighbor not in neighbors or depth < neighbors[neighbor]:
                        neighbors[neighbor] = depth
                
                if depth < max_depth:
                    queue.extend((neighbor, depth + 1) for neighbor in current_neighbors)
    
        return neighbors