# Analyze whether path lengths in the graph correlate with sentence similarity
# Ideally, there would be a strong negative correlation between path length and similarity
import argparse
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import stats
from src.graphs.top5 import load_graph
import random
from tqdm import tqdm
import itertools

def calculate_path_length(G, word1, word2):
    try:
        return nx.shortest_path_length(G, word1, word2)
    except nx.NetworkXNoPath:
        return float('inf')  # Return infinity if no path exists

def main(graph_path, sample_size):
    # Load the graph
    graph = load_graph(path=graph_path)

    # Convert the graph to a NetworkX graph for easier traversal
    G = nx.Graph(graph)

    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # List of words to compare
    keys = list(graph.keys())
    words = random.sample(keys, sample_size)

    # Calculate similarities
    word_pairs = list(itertools.combinations(words, 2))

    # Batch encode all words at once
    all_embeddings = model.encode(words)

    # Create a dictionary for fast embedding lookup
    embedding_dict = {word: embedding for word, embedding in zip(words, all_embeddings)}

    path_lengths = []
    sentence_similarities = []

    # Use tqdm for a progress bar
    for word1, word2 in tqdm(word_pairs, desc="Calculating similarities"):
        path_lengths.append(calculate_path_length(G, word1, word2))
        similarity = cosine_similarity([embedding_dict[word1]], [embedding_dict[word2]])[0][0]
        sentence_similarities.append(similarity)

    # Calculate summary statistics
    path_length_stats = {
        'mean': np.mean(path_lengths),
        'median': np.median(path_lengths),
        'std': np.std(path_lengths),
        'min': np.min(path_lengths),
        'max': np.max(path_lengths)
    }

    similarity_stats = {
        'mean': np.mean(sentence_similarities),
        'median': np.median(sentence_similarities),
        'std': np.std(sentence_similarities),
        'min': np.min(sentence_similarities),
        'max': np.max(sentence_similarities)
    }

    # Calculate correlations
    pearson_corr, pearson_p = stats.pearsonr(path_lengths, sentence_similarities)
    spearman_corr, spearman_p = stats.spearmanr(path_lengths, sentence_similarities)

    # Print summary statistics
    print("\nPath Length Statistics:")
    for key, value in path_length_stats.items():
        print(f"{key.capitalize()}: {value:.4f}")

    print("\nSentence Similarity Statistics:")
    for key, value in similarity_stats.items():
        print(f"{key.capitalize()}: {value:.4f}")

    print(f"\nPearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4f})")
    print(f"Spearman rank correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze path lengths and sentence similarities in a graph.")
    parser.add_argument("graph_path", type=str, help="Path to the word graph JSON file")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of words to sample (default: 100)")
    
    args = parser.parse_args()
    
    main(args.graph_path, args.sample_size)