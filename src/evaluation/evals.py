import pandas as pd

def scidocs(split):
    # Scidocs only has a validation and test split, so we consider train as validation, and validation as test
    if split not in ['train', 'test']:
        raise NotImplementedError("This dataset is not available for the split: " + split)
    
    splits = {'train': 'validation.jsonl.gz', 'test': 'test.jsonl.gz'}
    df = pd.read_json("hf://datasets/mteb/scidocs-reranking/" + splits[split], lines=True)
    return df