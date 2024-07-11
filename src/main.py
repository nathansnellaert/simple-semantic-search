import pandas as pd
from tqdm import tqdm
import json
import hashlib
import os
import argparse
from evaluation.metrics import get_reciprocal_rank
from evaluation.evals import scidocs
from graphs.top5 import load_graph
from sentence_transformers import SentenceTransformer
from similarity.sentence_transformers import SentenceTransformerCosine
from similarity.bm25 import BM25Similarity
from similarity.graph_traversal import GraphTraversalSimilarity
from similarity.jaccard import JaccardSimilarity

def calculate_mrr(df, similarity_function):
    mrr_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Calculating MRR"):
        query = row['query']
        positives = row['positive']
        negatives = row['negative']

        positive_similarities = [similarity_function.compute(query, pos) for pos in positives]
        negative_similarities = [similarity_function.compute(query, neg) for neg in negatives]

        rr = get_reciprocal_rank(positive_similarities, negative_similarities)
        mrr_scores.append(rr)

    return sum(mrr_scores) / len(mrr_scores)

def generate_id(dataset, similarity_function_name):
    unique_string = f"{dataset}_{similarity_function_name}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def save_results(results, dataset, similarity_function_name):
    unique_id = generate_id(dataset, similarity_function_name)
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{unique_id}.json")

    with open(output_path, 'w') as f:
        json.dump(results, f)


def main():
    parser = argparse.ArgumentParser(description="MRR Calculation")
    parser.add_argument("--dataset", type=str, default="scidocs", help="Dataset to use (default: scidocs)")
    parser.add_argument("--similarity_function", type=str, default="bm25", help="Similarity function to use (default: sentence-transformer)")
    args = parser.parse_args()

    dataset = args.dataset
    similarity_function_name = args.similarity_function

    if dataset == "scidocs":
        train_df, test_df = scidocs("train")[0:500], scidocs("validation")[0:500]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if similarity_function_name == "sentence-transformer":
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        similarity_function = SentenceTransformerCosine(model)

    elif similarity_function_name == "bm25":
        # Combine queries, positives, and negatives into the corpus
        corpus = train_df['query'].tolist() + \
                 [item for sublist in train_df['positive'].tolist() for item in sublist] + \
                 [item for sublist in train_df['negative'].tolist() for item in sublist]
        similarity_function = BM25Similarity(corpus)

    elif similarity_function_name == "graph":
        word_graph = load_graph()
        similarity_function = GraphTraversalSimilarity(word_graph)

    elif similarity_function_name == "jaccard":
        similarity_function = JaccardSimilarity()

    else:
        raise ValueError(f"Invalid similarity function name: {similarity_function_name}")

    mrr = calculate_mrr(test_df, similarity_function)
    results = {
        "dataset": dataset,
        "similarity_function": similarity_function_name,
        "mrr": mrr
    }

    save_results(results, dataset, similarity_function_name)
    print(f"Results saved to experiment_results/{generate_id(dataset, similarity_function_name)}.json")

if __name__ == "__main__":
    main()