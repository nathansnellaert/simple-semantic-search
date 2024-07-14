import itertools
import random
from main import run_experiment

def random_search(n_iterations=100):
    # Define parameter options
    datasets = ["scidocs"]
    similarity_functions = ["graph", "bm25", "jaccard", "sentence-transformer"]
    preprocessing_steps = ["lowercase", "remove_punctuation", "remove_numbers", "stem_words", "remove_stopwords", "remove_oov"]
    
    best_mrr = 0
    best_params = {}
    
    for _ in range(n_iterations):
        dataset = random.choice(datasets)
        sim_func = random.choice(similarity_functions)
        
        # Randomly select a number of preprocessing steps and shuffle them
        n_preproc = random.randint(1, len(preprocessing_steps))
        selected_preproc = random.sample(preprocessing_steps, n_preproc)
        preproc_str = ",".join(selected_preproc)
        
        results = run_experiment(dataset=dataset, similarity_function=sim_func, preprocessing=preproc_str)
        
        if results["mrr"] > best_mrr:
            best_mrr = results["mrr"]
            best_params = {
                "dataset": dataset,
                "similarity_function": sim_func,
                "preprocessing": preproc_str,
                "mrr": best_mrr
            }
        
        print(f"Params: {dataset}, {sim_func}, {preproc_str} - MRR: {results['mrr']}")
    
    print("\nBest parameters found:")
    print(best_params)

if __name__ == "__main__":   
    print("\nRunning Random Search:")
    random_search(n_iterations=50)