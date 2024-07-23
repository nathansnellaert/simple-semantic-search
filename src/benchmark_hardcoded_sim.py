import argparse
from benchmark import load_dataset, create_vocabulary, calculate_mrr, save_results
from similarity.hardcoded import HardcodedSimilarity
import time
import json

def run_experiment(dataset):
    train_df, test_df = load_dataset(dataset)
    vocabulary = create_vocabulary(train_df)

    similarity_function = HardcodedSimilarity()

    start_time = time.time()
    mrr_results = calculate_mrr(test_df, similarity_function, [], vocabulary)
    end_time = time.time()

    results = {
        "dataset": dataset,
        "similarity_function": "KnowledgeGraphSimilarity",
        "mrr_results": mrr_results,
        "execution_time": end_time - start_time
    }
    print(results)

    return results

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Similarity Benchmark")
    parser.add_argument("--dataset", type=str, default="scidocs", help="Dataset to use (default: scidocs)")
    args = parser.parse_args()

    results = run_experiment(
        args.dataset, 
    )
    
    save_results(results, "hardcoded sim")

if __name__ == "__main__":
    main()