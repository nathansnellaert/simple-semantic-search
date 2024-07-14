import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from tqdm import tqdm
import random
import itertools
import json
import matplotlib.pyplot as plt
import pandas as pd

def simple_pca(X, k):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    top_k_eigenvectors = eigenvectors[:, :k]
    return np.dot(X_centered, top_k_eigenvectors)

def load_graph(path):
    with open(path, 'r') as f:
        return json.load(f)

def to_int8(embeddings):
    # Scale to -128 to 127 range
    scaled = embeddings * 127
    return np.clip(np.round(scaled), -128, 127).astype(np.int8)

def from_int8(int8_embeddings):
    # Convert back to float, maintaining the -1 to 1 range
    return int8_embeddings.astype(float) / 127

def main(graph_path, sample_size):
    # Load the graph
    graph = load_graph(path=graph_path)

    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # List of words to compare
    keys = list(graph.keys())
    words = random.sample(keys, sample_size)

    # Calculate similarities
    word_pairs = list(itertools.combinations(words, 2))

    # Encode words
    embeddings = model.encode(words)

    # Dimensions to test
    dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]

    results = []

    for dim in tqdm(dimensions, desc="Processing dimensions"):
        # Perform dimensionality reduction
        reduced_embeddings = simple_pca(embeddings, dim)
        
        # Convert to int8 and back to float
        reduced_int8_embeddings = to_int8(reduced_embeddings)
        reduced_from_int8_embeddings = from_int8(reduced_int8_embeddings)

        original_similarities = []
        reduced_similarities = []
        reduced_int8_similarities = []

        for i, j in [(words.index(w1), words.index(w2)) for w1, w2 in word_pairs]:
            original_similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            reduced_similarity = cosine_similarity([reduced_embeddings[i]], [reduced_embeddings[j]])[0][0]
            reduced_int8_similarity = cosine_similarity([reduced_from_int8_embeddings[i]], [reduced_from_int8_embeddings[j]])[0][0]
            
            original_similarities.append(original_similarity)
            reduced_similarities.append(reduced_similarity)
            reduced_int8_similarities.append(reduced_int8_similarity)

        # Calculate correlations
        pearson_corr, pearson_p = stats.pearsonr(original_similarities, reduced_similarities)
        spearman_corr, spearman_p = stats.spearmanr(original_similarities, reduced_similarities)
        
        reduced_int8_pearson_corr, reduced_int8_pearson_p = stats.pearsonr(original_similarities, reduced_int8_similarities)
        reduced_int8_spearman_corr, reduced_int8_spearman_p = stats.spearmanr(original_similarities, reduced_int8_similarities)

        results.append({
            'dimension': dim,
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'reduced_int8_pearson_corr': reduced_int8_pearson_corr,
            'reduced_int8_pearson_p': reduced_int8_pearson_p,
            'reduced_int8_spearman_corr': reduced_int8_spearman_corr,
            'reduced_int8_spearman_p': reduced_int8_spearman_p
        })

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df['dimension'], df['pearson_corr'], label='Pearson Correlation (Float)')
    plt.plot(df['dimension'], df['spearman_corr'], label='Spearman Correlation (Float)')
    plt.plot(df['dimension'], df['reduced_int8_pearson_corr'], label='Pearson Correlation (Reduced Int8)')
    plt.plot(df['dimension'], df['reduced_int8_spearman_corr'], label='Spearman Correlation (Reduced Int8)')
    plt.xlabel('Dimension')
    plt.ylabel('Correlation')
    plt.title('Correlation between Original and Reduced/Int8 Embeddings')
    plt.legend()
    plt.xscale('log')  # Use log scale for x-axis
    plt.grid(True)

    # Save plot
    plt.savefig('correlation_plot.png')
    print("Plot saved as correlation_plot.png")

    # Save data
    df.to_csv('correlation_data.csv', index=False)
    print("Data saved as correlation_data.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze correlation between original, reduced, and reduced int8 word embeddings.")
    parser.add_argument("graph_path", type=str, help="Path to the word graph JSON file")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of words to sample (default: 100)")
    
    args = parser.parse_args()
    
    main(args.graph_path, args.sample_size)