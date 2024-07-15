import itertools
import concurrent.futures
import json
import os
import time
from benchmark_graph_search import run_experiment
from benchmark_bm25 import run_experiment as run_bm25_experiment
from benchmark_jaccard_sim import run_experiment as run_jaccard_experiment

def benchmark_graph_search():
    # Define parameter ranges for graph search
    datasets = ["scidocs"]
    max_depths = [1, 2]
    damping_factors = [0.5]
    preprocessing_steps = [
        "",
        "lowercase",
        "lowercase,remove_punctuation",
        "lowercase,remove_punctuation,remove_stopwords",
        "lowercase,remove_punctuation,remove_stopwords,stem_words",
        "lowercase,remove_punctuation,remove_stopwords,nltk_lemmatize",
        "lowercase,remove_punctuation,remove_stopwords,custom_lemmatize,nltk_lemmatize",
        "lowercase,remove_punctuation,remove_stopwords,custom_lemmatize,nltk_lemmatize,stem_words",
    ]
    graph_paths = [
        "./data/small_graph.json",
        "./data/word_graph.json",
    ]

    # Generate all combinations of parameters
    all_params = list(itertools.product(datasets, max_depths, damping_factors, preprocessing_steps, graph_paths))
    
    print(f"Total number of graph search experiments to run: {len(all_params)}")

    # Run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_experiment, all_params))

    return results

def run_single_experiment(params):
    dataset, max_depth, damping_factor, preprocessing, graph_path = params
    print(f"Running graph search experiment with params: {params}")
    results = run_experiment(dataset, max_depth, damping_factor, preprocessing, graph_path)
    print(f"Experiment complete: MRR = {results['mrr']}")
    return results

def benchmark_bm25():
    # Define parameter ranges for BM25
    datasets = ["scidocs"]
    k1_values = [1.2, 1.5, 1.8]
    b_values = [0.65, 0.75, 0.85]
    preprocessing_steps = [
        # "",
        # "lowercase",
        # "lowercase,remove_punctuation",
        "lowercase,remove_punctuation,remove_stopwords",
        "lowercase,remove_punctuation,remove_stopwords,stem_words",
        "lowercase,remove_punctuation,remove_stopwords,stem_words,nltk_lemmatize"
    ]

    # Generate all combinations of parameters
    all_params = list(itertools.product(datasets, k1_values, b_values, preprocessing_steps))
    
    print(f"Total number of BM25 experiments to run: {len(all_params)}")

    # Run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_bm25_experiment, all_params))

    return results

def run_single_bm25_experiment(params):
    dataset, k1, b, preprocessing = params
    print(f"Running BM25 experiment with params: {params}")
    results = run_bm25_experiment(dataset, k1, b, preprocessing)
    print(f"Experiment complete: MRR = {results['mrr']}")
    return results

def save_results(results, experiment_type):
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"{experiment_type}_{timestamp}.json")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

def benchmark_jaccard():
    # Define parameter ranges for Jaccard similarity
    datasets = ["scidocs"]
    preprocessing_steps = [
        "",
        "lowercase",
        "lowercase,remove_punctuation",
        "lowercase,remove_punctuation,remove_stopwords",
        "lowercase,remove_punctuation,remove_stopwords,stem_words",
        "lowercase,remove_punctuation,remove_stopwords,nltk_lemmatize",
        "lowercase,remove_punctuation,remove_stopwords,custom_lemmatize,nltk_lemmatize",
        "lowercase,remove_punctuation,remove_stopwords,custom_lemmatize,nltk_lemmatize,stem_words",
    ]

    # Generate all combinations of parameters
    all_params = list(itertools.product(datasets, preprocessing_steps))
    
    print(f"Total number of Jaccard similarity experiments to run: {len(all_params)}")

    # Run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_single_jaccard_experiment, all_params))

    return results

def run_single_jaccard_experiment(params):
    dataset, preprocessing = params
    print(f"Running Jaccard similarity experiment with params: {params}")
    results = run_jaccard_experiment(dataset, preprocessing)
    print(f"Experiment complete: MRR = {results['mrr']}")
    return results

def main():
    # Run graph search benchmarks
    # print("Starting Graph Search benchmarks...")
    # graph_search_results = benchmark_graph_search()
    # save_results(graph_search_results, "graph_search")

    # # Run BM25 benchmarks
    print("\nStarting BM25 benchmarks...")
    bm25_results = benchmark_bm25()
    save_results(bm25_results, "bm25")

    # Run Jaccard similarity benchmarks
    # print("\nStarting Jaccard similarity benchmarks...")
    # jaccard_results = benchmark_jaccard()
    # save_results(jaccard_results, "jaccard")


if __name__ == "__main__":
    main()