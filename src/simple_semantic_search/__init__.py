import json
import gzip
import requests

def normalize_text(text):
    return ''.join(char.lower() for char in text if char.isalnum() or char.isspace())

def get_extended_neighbors(word_graph, word, max_depth):
    neighbors = {}
    queue = [(word, 0)]

    while queue:
        current_word, depth = queue.pop(0)
        if depth > max_depth or current_word in neighbors:
            continue

        neighbors[current_word] = depth
        if current_word in word_graph:
            for neighbor in word_graph[current_word]:
                queue.append((neighbor, depth + 1))

    return neighbors

def weighted_jaccard_similarity(set1, set2):
    intersection_weight = 0
    union_weight = 0

    for word, depth in set1.items():
        weight = 1 if depth == 0 else 0.4
        union_weight += weight
        if word in set2:
            intersection_weight += weight

    for word, depth in set2.items():
        if word not in set1:
            weight = 1 if depth == 0 else 0.4
            union_weight += weight

    return intersection_weight / union_weight

class Search:
    def __init__(self, word_graph, max_depth=3):
        self.word_graph = word_graph
        self.max_depth = max_depth
        self.index = []

    def index_documents(self, docs):
        self.index = [
            {
                'text': doc,
                'words': {word: 0 for word in normalize_text(doc).split()}
            }
            for doc in docs
        ]

    def search(self, query, top_k=5):
        query_words = normalize_text(query).split()
        extended_query_words = {}
        
        for word in query_words:
            matches = self.find_partial_matches(word)
            for match in matches:
                extended_query_words[match] = 0
                neighbors = get_extended_neighbors(self.word_graph, match, self.max_depth)
                for neighbor, depth in neighbors.items():
                    if neighbor not in extended_query_words or extended_query_words[neighbor] > depth:
                        extended_query_words[neighbor] = depth
        
        scores = [
            {
                'doc': doc['text'],
                'score': weighted_jaccard_similarity(extended_query_words, doc['words'])
            }
            for doc in self.index
        ]

        scores.sort(key=lambda x: x['score'], reverse=True)
        return [result['doc'] for result in scores[:top_k]]

    def find_partial_matches(self, partial_word):
        matches = set()
        for doc in self.index:
            for word in doc['words']:
                if partial_word in word:
                    matches.add(word)
        return matches

def load(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"HTTP error! status: {response.status_code}")
    
    compressed_data = response.content
    decompressed_data = gzip.decompress(compressed_data)
    compressed_graph = json.loads(decompressed_data.decode('utf-8'))

    strings = compressed_graph['strings']
    
    decompressed_graph = {
        strings[int(k)]: [strings[i] for i in v]
        for k, v in compressed_graph['graph'].items()
    }
    
    return Search(decompressed_graph)