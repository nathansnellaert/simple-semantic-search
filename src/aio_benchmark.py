import itertools
import concurrent.futures
import json
import os
import time
from benchmark_graph_search import run_experiment as run_graph_search_experiment
from benchmark_bm25 import run_experiment as run_bm25_experiment
from benchmark_jaccard_sim import run_experiment as run_jaccard_experiment
from benchmark_sentence_transformers import run_experiment as run_sentence_transformers_experiment

# Set the dataset at the script level
DATASET = "scidocs"

def generate_params(param_ranges):
    return list(itertools.product(*param_ranges.values()))

def run_single_experiment(args):
    experiment_type, params, run_func = args
    print(f"Running {experiment_type} experiment with params: {params}")
    return run_func(*params)

def run_experiments(experiment_type, params, run_func):
    print(f"Total number of {experiment_type} experiments to run: {len(params)}")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = [(experiment_type, p, run_func) for p in params]
        results = list(executor.map(run_single_experiment, args))
    return results

def save_results(results, experiment_type):
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    output_path = os.path.join(output_dir, f"{experiment_type}_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

def benchmark_experiment(experiment_type, param_ranges, run_func):
    params = generate_params(param_ranges)
    results = run_experiments(experiment_type, params, run_func)
    save_results(results, experiment_type)
    return results

def get_best_config(results):
    if not results:
        return "N/A"
    best_result = max(results, key=lambda x: x['mrr_results']['average_mrr'])
    return str({k: v for k, v in best_result.items() if k != 'mrr_results'})

def generate_benchmark_md(experiments):
    table_rows = []
    for exp_type, config in experiments.items():
        results = config.get('results', [])
        if results:
            best_result = max(results, key=lambda x: x['mrr_results']['average_mrr'])
            best_mrr = best_result['mrr_results']['average_mrr']
            mean_mrr = sum(r['mrr_results']['average_mrr'] for r in results) / len(results)
            best_config = get_best_config(results)
        else:
            best_mrr, mean_mrr, best_config = 0, 0, "N/A"
        
        table_rows.append(f"| {exp_type} | {best_mrr:.3f} | {mean_mrr:.3f} | {best_config} |")

    table_content = "\n".join([
        "| Method | Best MRR | Mean MRR | Best Configuration |",
        "|--------|----------|----------|---------------------|",
        *table_rows
    ])

    with open("benchmark.md", "w") as f:
        f.write(table_content)

    print("benchmark.md has been generated.")

def main():
    experiments = {
        "Graph Search": {
            "param_ranges": {
                "dataset": [DATASET],
                "max_depth": [1],
                "damping_factor": [0.5, 0.8],
                "preprocessing": [
                    ["lowercase", "remove_punctuation"],
                    ["lowercase", "remove_punctuation", "remove_stopwords"],
                    ["lowercase", "remove_punctuation", "remove_stopwords", "custom_lemmatize"],
                ],
                "graph_path": ["./data/small_graph.json", "./data/word_graph.json"],
            },
            "run_func": run_graph_search_experiment,
        },
        "BM25": {
            "param_ranges": {
                "dataset": [DATASET],
                "k1": [1.2, 1.5, 1.8],
                "b": [0.65, 0.75, 0.85],
                "preprocessing": [
                    ["lowercase", "remove_punctuation", "remove_stopwords"],
                    ["lowercase", "remove_punctuation", "remove_stopwords", "stem_words"],
                    ["lowercase", "remove_punctuation", "remove_stopwords", "stem_words", "nltk_lemmatize"],
                ],
            },
            "run_func": run_bm25_experiment,
        },
        "Jaccard": {
            "param_ranges": {
                "dataset": [DATASET],
                "preprocessing": [
                    [],
                    ["lowercase"],
                    ["lowercase", "remove_punctuation"],
                    ["lowercase", "remove_punctuation", "remove_stopwords"],
                    ["lowercase", "remove_punctuation", "remove_stopwords", "stem_words"],
                    ["lowercase", "remove_punctuation", "remove_stopwords", "stem_words", "nltk_lemmatize"],
                ],
            },
            "run_func": run_jaccard_experiment,
        },
        "Sentence Transformers": {
            "param_ranges": {
                "dataset": [DATASET],
                "model": [
                    "all-MiniLM-L6-v2", 
                    "all-mpnet-base-v2", 
                    "BAAI/bge-base-en-v1.5"
               ]
            },
            "run_func": run_sentence_transformers_experiment,
        },
    }

    for experiment_type, config in experiments.items():
        print(f"\nStarting {experiment_type} benchmarks...")
        results = benchmark_experiment(experiment_type, config["param_ranges"], config["run_func"])
        config['results'] = results

    generate_benchmark_md(experiments)

if __name__ == "__main__":
    main()