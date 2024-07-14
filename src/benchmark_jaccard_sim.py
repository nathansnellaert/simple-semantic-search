import argparse
from benchmark import load_dataset, create_vocabulary, calculate_mrr, save_results
from similarity.jaccard import JaccardSimilarity
import time

def run_experiment(dataset, preprocessing):
    preprocessing_steps = preprocessing.split(',') if preprocessing else []

    train_df, test_df = load_dataset(dataset)
    vocabulary = create_vocabulary(train_df)

    similarity_function = JaccardSimilarity()

    mrr, none_percentage = calculate_mrr(test_df, similarity_function, preprocessing_steps, vocabulary)

    results = {
        "dataset": dataset,
        "similarity_function": "JaccardSimilarity",
        "preprocessing_steps": preprocessing_steps,
        "mrr": mrr,
        "none_percentage": none_percentage
    }

    print(results)

    return results

def main():
    parser = argparse.ArgumentParser(description="Jaccard Similarity Benchmark")
    parser.add_argument("--dataset", type=str, default="scidocs", help="Dataset to use (default: scidocs)")
    parser.add_argument("--preprocessing", type=str, default="", help="Comma-separated list of preprocessing steps")
    args = parser.parse_args()

    results = run_experiment(args.dataset, args.preprocessing)

    id = time.time()
    save_results(results, id)
    print(f"Results saved to experiment_results/{id}.json")
    print(f"MRR: {results['mrr']}")
    print(f"Percentage of None values: {results['none_percentage']}%")

if __name__ == "__main__":
    main()