# Simple Semantic Search

Rule based search library, that leverages a graph of keywords match semantically. Outperforms BM25 on a variety of datasets. Currently under development, not production ready.

## Introduction

Keyword search offers speed and precision but lacks context. Semantic search understands intent but can be complex and resource-intensive. Simple Semantic Search aims to sit in the middle, offering a lightweight solution that has pretty good performance.

## Installation

```bash
pip install simple-semantic-search
```

# Quickstart
```python
from simple_semantic_search import load

# load the desired graph
engine = load("https://raw.githubusercontent.com/nathansnellaert/simple-semantic-search-js/main/data/graph.json.gz")

products = [
    "Wireless noise-cancelling headphones with Bluetooth",
    "Ergonomic office chair with lumbar support",
    "Stainless steel water bottle, vacuum insulated",
    "Smart LED bulb, color changing, WiFi enabled",
]

engine.index_documents(products)
results = engine.search("cordless headset for work", top_k=3)

print(results)
```
