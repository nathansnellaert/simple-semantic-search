# Storing embeddings

Sentence embeddings are typically generated on-demand, which is not feasible for client-side web applications. However, we might be able to leverage stored embeddings for a simple semantic search library if:

1. **Word Embedding Aggregation**: We can store embeddings for individual words/tokens instead of full sentences. If bag-of-words representations correlate highly with full sentence embeddings, we could generate approximate sentence embeddings on-the-fly from these stored word embeddings.

2. **Dimensionality Reduction**: By applying dimensionality reduction to our embeddings, we can preserve most of the relevant information while significantly reducing the data that needs to be transferred to the client.

3. **Quantization**: Using 8-bit integers instead of 32-bit floats can further reduce data size with minimal information loss.

Example storage calculation:
- 50,000 tokens (a reasonable vocabulary for many domains)
- 10 dimensions per embedding (after reduction)
- 8-bit integer per dimension

Data size: 50,000 * 10 * 1 byte = 500,000 bytes ≈ 488.28 KB