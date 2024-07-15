# should properly split up these functions, some of this is pretty core to the engine, not just benchmarking.
import pandas as pd
from tqdm import tqdm
import json
import os
import time
from preprocessing.numbers import remove_numbers
from preprocessing.stemming import stem_words
from preprocessing.lemmatizer import nltk_lemmatization, custom_graph_lemmatization, CustomGraphLemmatizer
from preprocessing.stopwords import nltk_stopword_removal
from preprocessing.simple import lowercase, remove_punctuation, remove_oov
from evaluation.metrics import get_reciprocal_rank
from evaluation.evals import scidocs, custom

# Initialize the custom lemmatizer
graph_path = './data/lemma_mapping.json'
custom_lemmatizer = CustomGraphLemmatizer(graph_path)

preprocessing_functions = {
    'lowercase': lowercase,
    'remove_punctuation': remove_punctuation,
    'remove_numbers': remove_numbers,
    'stem_words': stem_words,
    'remove_stopwords': nltk_stopword_removal,
    'nltk_lemmatize': nltk_lemmatization,
    'custom_lemmatize': lambda text: custom_graph_lemmatization(text, custom_lemmatizer),
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
            mrr_scores.append(0)
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

def load_dataset(dataset):
    if dataset == "scidocs":
        train_df, test_df = scidocs("train"), scidocs("test")[0:500]
    elif dataset == "custom":
        train_df, test_df = custom("train"), custom("test")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return train_df, test_df