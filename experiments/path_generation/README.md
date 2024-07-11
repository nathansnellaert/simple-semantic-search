### Path generation

The existing path crawler works, but it is unsupervised. 

If we consider sentencetransformers the gold standard, we could consider similarity scores between embeddings as labels.

One way to leverage that is to ask a language model to generate paths between two nodes, where the path length is based on the similarity score. 

Though it remains to be seen if this is more effective than path generation similarity based counts. 