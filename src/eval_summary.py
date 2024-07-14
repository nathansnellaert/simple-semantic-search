import os
import json
import pandas as pd

def analyze_results():
    results_dir = "experiment_results"
    all_results = []

    # Read all result files
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            with open(os.path.join(results_dir, filename), 'r') as f:
                result = json.load(f)
                all_results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Get the highest ranked one for each similarity function
    best_results = df.groupby('similarity_function').apply(lambda x: x.loc[x['mrr'].idxmax()])

    print("All results:")
    print(df)
    print("\nBest results for each similarity function:")
    print(best_results)

    return df, best_results

if __name__ == "__main__":
    all_results_df, best_results_df = analyze_results()