import argparse
from benchmark import load_dataset, create_vocabulary, calculate_mrr, save_results
from sentence_transformers import SentenceTransformer
from src.similarity.sentence_transformers import SentenceTransformerCosine
import time

def run_experiment(dataset, model_name):
    train_df, test_df = load_dataset(dataset)
    vocabulary = create_vocabulary(train_df)

    model = SentenceTransformer(model_name)
    similarity_function = SentenceTransformerCosine(model)

    mrr_results = calculate_mrr(test_df, similarity_function, preprocessing_steps=[], vocabulary=vocabulary)

    results = {
        "dataset": dataset,
        "similarity_function": f"SentenceTransformer({model_name})",
        "model_name": model_name,
        "mrr_results": mrr_results
    }

    return results

def main():
    parser = argparse.ArgumentParser(description="Sentence Transformers Benchmark")
    parser.add_argument("--dataset", type=str, default="scidocs", help="Dataset to use (default: scidocs)")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model to use (default: all-MiniLM-L6-v2)")
    args = parser.parse_args()

    results = run_experiment(args.dataset, args.model)

    id = time.time()
    save_results(results, id)
    print(f"Results saved to experiment_results/{id}.json")
    print(f"MRR: {results['mrr_results']['mrr']}")
    print(f"Percentage of None values: {results['mrr_results']['none_percentage']}%")

if __name__ == "__main__":
    main()