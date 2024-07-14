# Main script to run experiments. 
# TODO: allow specification of similarity function specific parameters 
import pandas as pd
from tqdm import tqdm
import json
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
import time
from preprocessing.numbers import remove_numbers
from preprocessing.stemming import stem_words
from preprocessing.lemmatizer import nltk_lemmatization
from preprocessing.stopwords import nltk_stopword_removal
from preprocessing.simple import lowercase, remove_punctuation, remove_oov


preprocessing_functions = {
    'lowercase': lowercase,
    'remove_punctuation': remove_punctuation,
    'remove_numbers': remove_numbers,
    'stem_words': stem_words,
    'remove_stopwords': nltk_stopword_removal,
    'lemmatize': nltk_lemmatization,
    'remove_oov': remove_oov
}

def apply_preprocessing(text, preprocessing_steps, vocabulary):
    for step in preprocessing_steps:
        if step == 'remove_oov':
            text = preprocessing_functions[step](text, vocabulary)
        else:
            text = preprocessing_functions[step](text)
    return text

def calculate_mrr(df, similarity_function, preprocessing_steps, vocabulary):
    mrr_scores = []
    none_count = 0
    total_count = len(df)
    
    for _, row in tqdm(df.iterrows(), total=total_count, desc="Calculating MRR"):
        query = apply_preprocessing(row['query'], preprocessing_steps, vocabulary)
        positives = [apply_preprocessing(pos, preprocessing_steps, vocabulary) for pos in row['positive']]
        negatives = [apply_preprocessing(neg, preprocessing_steps, vocabulary) for neg in row['negative']]

        positive_similarities = [similarity_function.compute(query, pos) for pos in positives]
        negative_similarities = [similarity_function.compute(query, neg) for neg in negatives]

        rr = get_reciprocal_rank(positive_similarities, negative_similarities)
        if rr is None:
            none_count += 1
        else:
            mrr_scores.append(rr)

    valid_scores = [score for score in mrr_scores if score is not None]
    average_mrr = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    none_percentage = (none_count / total_count) * 100

    return average_mrr, none_percentage

def save_results(results, id):
    output_dir = "experiment_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{id}.json")

    with open(output_path, 'w') as f:
        json.dump(results, f)

def create_vocabulary(df):
    vocabulary = set()
    vocabulary.update(df['query'].str.split().explode())
    vocabulary.update(df['positive'].explode().str.split().explode())
    vocabulary.update(df['negative'].explode().str.split().explode())
    return vocabulary


def run_experiment(dataset="scidocs", similarity_function="bm25", preprocessing=""):
    preprocessing_steps = preprocessing.split(',') if preprocessing else []

    if dataset == "scidocs":
        train_df, test_df = scidocs("train"), scidocs("test")[0:500]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Always create vocabulary
    vocabulary = create_vocabulary(train_df)

    if similarity_function == "sentence-transformer":
        model = SentenceTransformer('all-MiniLM-L6-v2')
        similarity_function = SentenceTransformerCosine(model)
    elif similarity_function == "bm25":
        corpus = train_df['query'].tolist() + \
                 [item for sublist in train_df['positive'].tolist() for item in sublist] + \
                 [item for sublist in train_df['negative'].tolist() for item in sublist]
        corpus = [apply_preprocessing(text, preprocessing_steps, vocabulary) for text in corpus]
        similarity_function = BM25Similarity(corpus)
    elif similarity_function == "graph":
        word_graph = load_graph()
        similarity_function = GraphTraversalSimilarity(word_graph, max_depth=3)
    elif similarity_function == "jaccard":
        similarity_function = JaccardSimilarity()
    else:
        raise ValueError(f"Invalid similarity function name: {similarity_function}")

    mrr, none_percentage = calculate_mrr(test_df, similarity_function, preprocessing_steps, vocabulary)

    results = {
        "dataset": dataset,
        "similarity_function": similarity_function.__class__.__name__,
        "preprocessing_steps": preprocessing_steps,
        "mrr": mrr,
        "none_percentage": none_percentage
    }

    id = time.time()
    save_results(results, id)
    print(f"Results saved to experiment_results/{id}.json")
    print(f"MRR: {mrr}")
    print(f"Percentage of None values: {none_percentage}%")
    return results

def main():
    parser = argparse.ArgumentParser(description="MRR Calculation")
    parser.add_argument("--dataset", type=str, default="scidocs", help="Dataset to use (default: scidocs)")
    parser.add_argument("--similarity_function", type=str, default="bm25", help="Similarity function to use (default: bm25)")
    parser.add_argument("--preprocessing", type=str, default="", help="Comma-separated list of preprocessing steps")
    args = parser.parse_args()

    run_experiment(args.dataset, args.similarity_function, args.preprocessing)

if __name__ == "__main__":
    main()