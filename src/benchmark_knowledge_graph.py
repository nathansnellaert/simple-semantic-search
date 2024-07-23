import argparse
from benchmark import load_dataset, create_vocabulary, calculate_mrr, save_results
from similarity.knowledge_graph import KnowledgeGraphSimilarity
import time
import json

def run_experiment(dataset, max_depth, damping_factor, use_weights, use_directionality, 
                   use_relationship_similarity, preprocessing, graph_path):
    preprocessing_steps = preprocessing if isinstance(preprocessing, list) else preprocessing.split(',')

    train_df, test_df = load_dataset(dataset)
    vocabulary = create_vocabulary(train_df)

    with open(graph_path, 'r') as f:
        word_graph = json.load(f)

    similarity_function = KnowledgeGraphSimilarity(
        word_graph, 
        max_depth=max_depth, 
        damping_factor=damping_factor,
        use_weights=use_weights,
        use_directionality=use_directionality,
        use_relationship_similarity=use_relationship_similarity
    )

    start_time = time.time()
    mrr_results = calculate_mrr(test_df, similarity_function, preprocessing_steps, vocabulary)
    end_time = time.time()

    results = {
        "dataset": dataset,
        "similarity_function": "KnowledgeGraphSimilarity",
        "preprocessing_steps": preprocessing_steps,
        "max_depth": max_depth,
        "damping_factor": damping_factor,
        "use_weights": use_weights,
        "use_directionality": use_directionality,
        "use_relationship_similarity": use_relationship_similarity,
        "graph_path": graph_path,
        "mrr_results": mrr_results,
        "execution_time": end_time - start_time
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Similarity Benchmark")
    parser.add_argument("--dataset", type=str, default="scidocs", help="Dataset to use (default: scidocs)")
    parser.add_argument("--max_depth", type=int, default=3, help="Maximum depth for graph traversal (default: 3)")
    parser.add_argument("--damping_factor", type=float, default=0.85, help="Damping factor for graph traversal (default: 0.85)")
    parser.add_argument("--use_weights", type=bool, default=True, help="Use relationship weights (default: True)")
    parser.add_argument("--use_directionality", type=bool, default=True, help="Use directional relationships (default: True)")
    parser.add_argument("--use_relationship_similarity", type=bool, default=True, help="Use relationship-specific similarity (default: True)")
    parser.add_argument("--preprocessing", type=str, default="", help="Comma-separated list of preprocessing steps")
    parser.add_argument("--graph_path", type=str, default="./data/knowledge_graph.json", help="Path to the knowledge graph JSON file")
    args = parser.parse_args()

    results = run_experiment(
        args.dataset, 
        args.max_depth, 
        args.damping_factor, 
        args.use_weights,
        args.use_directionality,
        args.use_relationship_similarity,
        args.preprocessing,
        args.graph_path
    )
    
    save_results(results, "knowledge_graph_similarity")

if __name__ == "__main__":
    main()