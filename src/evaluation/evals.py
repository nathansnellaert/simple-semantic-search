import pandas as pd
import os
import math

def scidocs(split):
    # Scidocs only has a validation and test split, so we consider train as validation, and validation as test
    if split not in ['train', 'test']:
        raise NotImplementedError("This dataset is not available for the split: " + split)
    
    splits = {'train': 'validation.jsonl.gz', 'test': 'test.jsonl.gz'}
    df = pd.read_json("hf://datasets/mteb/scidocs-reranking/" + splits[split], lines=True)
    df['file_id'] = "scidocs"
    return df

# for now use a simple 50/50 split
def custom(split):
    if split not in ['train', 'test']:
        raise NotImplementedError("This dataset is only available for 'train' and 'test' splits")

    dir = "./labels"
    files = [f for f in os.listdir(dir) if f.endswith('.json')]
    dfs = []
    for f in files:
        df = pd.read_json(os.path.join(dir, f))
        file_id = os.path.splitext(f)[0]
        df['file_id'] = file_id
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df['positive'] = df['positive'].apply(lambda x: [x])

    # Calculate the split point
    total_rows = len(df)
    split_point = math.ceil(total_rows / 2)

    # Split the data
    if split == 'train':
        df = df.iloc[:split_point]
    else:  # test
        df = df.iloc[split_point:]

    return df