import argparse
from benchmark import load_dataset, create_vocabulary, calculate_mrr, save_results
from graphs.top5 import load_graph
from similarity.graph_traversal import GraphTraversalSimilarity
import time

def run_experiment(dataset, max_depth, damping_factor, preprocessing, graph_path):
    preprocessing_steps = preprocessing.split(',') if preprocessing else []

    train_df, test_df = load_dataset(dataset)
    vocabulary = create_vocabulary(train_df)

    word_graph = load_graph(graph_path)
    similarity_function = GraphTraversalSimilarity(word_graph, max_depth=max_depth, damping_factor=damping_factor)

    mrr, none_percentage = calculate_mrr(test_df, similarity_function, preprocessing_steps, vocabulary)

    results = {
        "dataset": dataset,
        "similarity_function": "GraphTraversalSimilarity",
        "preprocessing_steps": preprocessing_steps,
        "max_depth": max_depth,
        "damping_factor": damping_factor,
        "mrr": mrr,
        "none_percentage": none_percentage,
        "graph_path": graph_path  
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="Graph Traversal Similarity Benchmark")
    parser.add_argument("--dataset", type=str, default="scidocs", help="Dataset to use (default: scidocs)")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth for graph traversal (default: 3)")
    parser.add_argument("--damping_factor", type=float, default=0.85, help="Damping factor for graph traversal (default: 0.85)")
    parser.add_argument("--preprocessing", type=str, default="", help="Comma-separated list of preprocessing steps")
    args = parser.parse_args()

    run_experiment(args.dataset, args.max_depth, args.damping_factor, args.preprocessing)

if __name__ == "__main__":
    main()