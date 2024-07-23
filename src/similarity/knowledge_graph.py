from .base import Similarity
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class KnowledgeGraphSimilarity(Similarity):
    DEFAULT_RELATIONSHIP_WEIGHTS = {
        "is_a": 1.5, "part_of": 1.3, "has_property": 1.2, "can": 1.1,
        "synonym": 1.8, "antonym": 0.5, "causes": 1.2, "used_for": 1.1,
        "located_in": 1.0, "time_related": 0.9, "made_of": 1.2, "produces": 1.1,
        "requires": 1.0, "instance_of": 1.4, "associated_with": 1.0,
        "derived_from": 1.1, "measures": 0.9, "opposite_of": 0.6,
        "symbolizes": 0.8, "precedes": 0.9
    }

    def __init__(self, word_graph: List[Dict[str, str]], max_depth: int, 
                 damping_factor: float = 0.85, 
                 use_weights: bool = True,
                 use_directionality: bool = True,
                 use_relationship_similarity: bool = True,
                 custom_weights: Optional[Dict[str, float]] = None):
        self.max_depth = max_depth
        self.damping_factor = damping_factor
        self.use_weights = use_weights
        self.use_directionality = use_directionality
        self.use_relationship_similarity = use_relationship_similarity
        self.relationship_weights = custom_weights if custom_weights else self.DEFAULT_RELATIONSHIP_WEIGHTS
        self.word_graph = self._build_graph(word_graph)

    def _build_graph(self, word_graph: List[Dict[str, str]]) -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
        graph = defaultdict(lambda: {'outgoing': [], 'incoming': []})
        for entry in word_graph:
            if self.use_directionality:
                graph[entry['word1']]['outgoing'].append((entry['word2'], entry['relation']))
                graph[entry['word2']]['incoming'].append((entry['word1'], entry['relation']))
            else:
                graph[entry['word1']]['outgoing'].append((entry['word2'], entry['relation']))
                graph[entry['word2']]['outgoing'].append((entry['word1'], entry['relation']))
        return graph

    def compute(self, phrase1: str, phrase2: str) -> float:
        words1 = set(phrase1.split())
        words2 = set(phrase2.split())
        
        direct_matches = words1.intersection(words2)
        direct_match_score = len(direct_matches) / max(len(words1), len(words2))
        
        remaining_words1 = words1 - direct_matches
        remaining_words2 = words2 - direct_matches
        
        traversal1 = self._calculate_traversal(remaining_words1)
        traversal2 = self._calculate_traversal(remaining_words2)
        
        graph_similarity = self._calculate_graph_similarity(traversal1, traversal2)
        
        final_similarity = (direct_match_score + graph_similarity) / 2
        
        return final_similarity

    def _calculate_traversal(self, words: set) -> Dict[str, Dict[str, Dict[str, float]]]:
        traversal_scores = defaultdict(lambda: {'outgoing': defaultdict(float), 'incoming': defaultdict(float)})
        
        for word in words:
            directions = ['outgoing', 'incoming'] if self.use_directionality else ['outgoing']
            for direction in directions:
                neighbors = self._get_extended_neighbors(word, self.max_depth, direction)
                
                for neighbor, depth, relation in neighbors:
                    if self.use_weights:
                        relation_weight = self.relationship_weights.get(relation, 1.0)
                        neighbor_score = (self.damping_factor ** depth) * relation_weight
                    else:
                        neighbor_score = self.damping_factor ** depth
                    
                    if self.use_relationship_similarity:
                        traversal_scores[neighbor][direction][relation] += neighbor_score
                    else:
                        traversal_scores[neighbor][direction]['generic'] += neighbor_score
        
        return traversal_scores
    
    def _get_extended_neighbors(self, word: str, max_depth: int, direction: str) -> List[Tuple[str, int, str]]:
        neighbors = []
        queue = [(word, 0, None)]
        visited = set()

        while queue:
            current_word, depth, relation = queue.pop(0)
            if depth > max_depth:
                break
            
            if current_word not in visited:
                visited.add(current_word)
                if depth > 0:
                    neighbors.append((current_word, depth, relation))
                
                if depth < max_depth:
                    for neighbor, rel in self.word_graph.get(current_word, {}).get(direction, []):
                        queue.append((neighbor, depth + 1, rel))
    
        return neighbors

    def _calculate_graph_similarity(self, traversal1: Dict[str, Dict[str, Dict[str, float]]], 
                                    traversal2: Dict[str, Dict[str, Dict[str, float]]]) -> float:
        all_words = set(traversal1.keys()) | set(traversal2.keys())
        directions = ['outgoing', 'incoming'] if self.use_directionality else ['outgoing']
        relations = set(self.relationship_weights.keys()) if self.use_relationship_similarity else ['generic']
        
        total_similarity = 0
        total_weight = 0
        
        for word in all_words:
            word_similarity = 0
            word_weight = 0
            
            for direction in directions:
                for relation in relations:
                    score1 = traversal1.get(word, {}).get(direction, {}).get(relation, 0)
                    score2 = traversal2.get(word, {}).get(direction, {}).get(relation, 0)
                    
                    if score1 > 0 or score2 > 0:
                        relation_similarity = min(score1, score2) / max(score1, score2)
                        relation_weight = self.relationship_weights.get(relation, 1.0) if self.use_weights else 1.0
                        
                        word_similarity += relation_similarity * relation_weight
                        word_weight += relation_weight
            
            if word_weight > 0:
                total_similarity += word_similarity / word_weight
                total_weight += 1
        
        return total_similarity / total_weight if total_weight > 0 else 0

    def get_relationship_weight(self, relationship: str) -> float:
        return self.relationship_weights.get(relationship, 1.0)

    def set_relationship_weight(self, relationship: str, weight: float):
        self.relationship_weights[relationship] = weight

    def set_use_weights(self, use_weights: bool):
        self.use_weights = use_weights

    def set_use_directionality(self, use_directionality: bool):
        self.use_directionality = use_directionality

    def set_use_relationship_similarity(self, use_relationship_similarity: bool):
        self.use_relationship_similarity = use_relationship_similarity