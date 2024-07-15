import argparse
from benchmark import load_dataset, create_vocabulary, calculate_mrr, save_results
from similarity.jaccard import JaccardSimilarity
import time

def run_experiment(dataset, preprocessing):
    train_df, test_df = load_dataset(dataset)
    vocabulary = create_vocabulary(train_df)

    similarity_function = JaccardSimilarity()

    mrr_results = calculate_mrr(test_df, similarity_function, preprocessing, vocabulary)

    results = {
        "dataset": dataset,
        "similarity_function": "JaccardSimilarity",
        "preprocessing_steps": preprocessing,
        "mrr_results": mrr_results
    }

    print(results)

    return results

def main():
    parser = argparse.ArgumentParser(description="Jaccard Similarity Benchmark")
    parser.add_argument("--dataset", type=str, default="scidocs", help="Dataset to use (default: scidocs)")
    parser.add_argument("--preprocessing", nargs='+', default=[], help="List of preprocessing steps")
    args = parser.parse_args()

    results = run_experiment(args.dataset, args.preprocessing)

    id = time.time()
    save_results(results, id)
    print(f"Results saved to experiment_results/{id}.json")
    print(f"MRR: {results['mrr_results']['mrr']}")
    print(f"Percentage of None values: {results['mrr_results']['none_percentage']}%")

if __name__ == "__main__":
    main()