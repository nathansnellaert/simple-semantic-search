import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from src.evaluation.evals import scidocs

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_sentence(sentence):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(sentence.lower())
    return ' '.join([token for token in tokens if token.isalnum() and token not in stop_words])

def create_vocabulary(train_df, test_df):
    vocabulary = set()
    for df in [train_df, test_df]:
        vocabulary.update(df['query'].str.split().explode())
        vocabulary.update(df['positive'].explode().str.split().explode())
        vocabulary.update(df['negative'].explode().str.split().explode())
    return vocabulary

def tfidf_weighted_embedding(tfidf_vector, word_embeddings, feature_names):
    weighted_embeddings = []
    for word, tfidf_score in zip(feature_names, tfidf_vector):
        if word in word_embeddings and tfidf_score > 0:
            weighted_embeddings.append(word_embeddings[word] * tfidf_score)
    return np.mean(weighted_embeddings, axis=0) if weighted_embeddings else None

def unweighted_embedding(query, word_embeddings):
    embeddings = [word_embeddings[word] for word in query.split() if word in word_embeddings]
    return np.mean(embeddings, axis=0) if embeddings else None

def safe_correlation(a, b):
    if a is None or b is None:
        return None
    if np.all(a == 0) or np.all(b == 0):
        return 0
    return np.corrcoef(a, b)[0, 1]

def plot_histogram(correlations, title, filename):
    plt.figure(figsize=(10, 6))
    plt.hist([c for c in correlations if c is not None], bins=10, edgecolor='black')
    avg_correlation = np.mean([c for c in correlations if c is not None])
    plt.title(title)
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.axvline(avg_correlation, color='r', linestyle='dashed', linewidth=2, label=f'Mean ({avg_correlation:.4f})')
    plt.legend()
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")
    plt.close()

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')

    train, test = scidocs("train"), scidocs("test")

    # Create full vocabulary
    vocabulary = create_vocabulary(train, test)

    # Preprocess all queries
    train_queries = [preprocess_sentence(query) for query in train['query'].tolist()]
    test_queries = [preprocess_sentence(query) for query in test['query'].tolist()]
    all_queries = train_queries + test_queries

    # Create TF-IDF vectorizer with full vocabulary
    tfidf = TfidfVectorizer(vocabulary=vocabulary, lowercase=False)
    tfidf.fit(all_queries)
    tfidf_matrix = tfidf.transform(test_queries)
    feature_names = tfidf.get_feature_names_out()

    # Compute word embeddings only for words in test queries
    unique_words = set(word for query in test_queries for word in query.split())
    word_embeddings = {word: model.encode([word])[0] for word in tqdm(unique_words, desc="Computing word embeddings")}

    tfidf_correlations = []
    unweighted_correlations = []

    for query, tfidf_vector in tqdm(zip(test_queries, tfidf_matrix), total=len(test_queries), desc="Processing queries"):
        # Get full query embedding
        full_embedding = model.encode([query])[0]
        
        # Get TF-IDF weighted bag-of-words embedding
        tfidf_bow_embedding = tfidf_weighted_embedding(tfidf_vector.toarray()[0], word_embeddings, feature_names)
        
        # Get unweighted bag-of-words embedding
        unweighted_bow_embedding = unweighted_embedding(query, word_embeddings)
        
        # Calculate correlations
        tfidf_correlation = safe_correlation(full_embedding, tfidf_bow_embedding)
        unweighted_correlation = safe_correlation(full_embedding, unweighted_bow_embedding)
        
        tfidf_correlations.append(tfidf_correlation)
        unweighted_correlations.append(unweighted_correlation)

    # Calculate average correlations
    avg_tfidf_correlation = np.mean([c for c in tfidf_correlations if c is not None])
    avg_unweighted_correlation = np.mean([c for c in unweighted_correlations if c is not None])
    
    print(f"Average TF-IDF weighted correlation: {avg_tfidf_correlation:.4f}")
    print(f"Average unweighted correlation: {avg_unweighted_correlation:.4f}")

    # Plot histograms
    plot_histogram(tfidf_correlations, 'Distribution of TF-IDF Weighted Correlations for Queries', 'tfidf_correlation_histogram.png')
    plot_histogram(unweighted_correlations, 'Distribution of Unweighted Correlations for Queries', 'unweighted_correlation_histogram.png')

if __name__ == "__main__":
    main()