import argparse
from benchmark import load_dataset, create_vocabulary, calculate_mrr, save_results
from similarity.bm25 import BM25Similarity
import time

def run_experiment(dataset, k1, b, preprocessing):
    preprocessing_steps = preprocessing.split(',') if preprocessing else []

    train_df, test_df = load_dataset(dataset)
    vocabulary = create_vocabulary(train_df)

    # Create corpus from training data
    corpus = train_df['query'].tolist() + \
             [item for sublist in train_df['positive'].tolist() for item in sublist] + \
             [item for sublist in train_df['negative'].tolist() for item in sublist]

    similarity_function = BM25Similarity(corpus, k1=k1, b=b)

    mrr, none_percentage = calculate_mrr(test_df, similarity_function, preprocessing_steps, vocabulary)

    results = {
        "dataset": dataset,
        "similarity_function": "BM25Similarity",
        "preprocessing_steps": preprocessing_steps,
        "k1": k1,
        "b": b,
        "mrr": mrr,
        "none_percentage": none_percentage
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="BM25 Similarity Benchmark")
    parser.add_argument("--dataset", type=str, default="scidocs", help="Dataset to use (default: scidocs)")
    parser.add_argument("--k1", type=float, default=1.5, help="k1 parameter for BM25 (default: 1.5)")
    parser.add_argument("--b", type=float, default=0.75, help="b parameter for BM25 (default: 0.75)")
    parser.add_argument("--preprocessing", type=str, default="", help="Comma-separated list of preprocessing steps")
    args = parser.parse_args()

    results = run_experiment(args.dataset, args.k1, args.b, args.preprocessing)

    id = time.time()
    save_results(results, id)
    print(f"Results saved to experiment_results/{id}.json")
    print(f"MRR: {results['mrr']}")
    print(f"Percentage of None values: {results['none_percentage']}%")

if __name__ == "__main__":
    main()