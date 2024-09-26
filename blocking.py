import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
import string
import numpy as np
import time

def create_block_index(csv_file_path, stopwords_file_path, device='cuda:0', min_word_length=3, min_word_count=1):
    # Start the timer
    block_index_start = time.time()

    # Determine device: CUDA or CPU
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Load stopwords from the file
    with open(stopwords_file_path, "r") as file:
        stopwords = file.read().splitlines()
    stopwords = stopwords + ['nan', 'NaN']

    # Concatenate text columns
    texts = df.iloc[:, 1:].astype(str).apply(lambda x: '>'.join(x), axis=1).tolist()

    # Remove punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    texts_without_punctuation = [text.translate(translator) for text in texts]

    # Tokenizer that filters words based on length
    def custom_tokenizer(text):
        words = [word for word in text.split() if len(word) >= min_word_length]
        return words

    # Create CountVectorizer instance with custom tokenizer and stopwords
    vectorizer = CountVectorizer(binary=True, tokenizer=custom_tokenizer, stop_words=stopwords)
    
    # Transform texts into a sparse matrix
    X_sparse = vectorizer.fit_transform(texts_without_punctuation)

    # Convert to COO matrix format
    X_coo = coo_matrix(X_sparse)

    # Sum word occurrences and filter out rare words
    word_counts = X_sparse.sum(axis=0)
    valid_word_indices = (word_counts > min_word_count).flatten()
    valid_word_indices = np.array(valid_word_indices)[0]

    # Filter the sparse matrix with valid word indices
    X_sparse_filtered = X_sparse[:, valid_word_indices]
    X_coo_filtered = coo_matrix(X_sparse_filtered)

    # Extract indices and values for the sparse COO tensor
    indices = [X_coo_filtered.row.tolist(), X_coo_filtered.col.tolist()]
    values = X_coo_filtered.data.tolist()
    dense_shape = tuple((X_coo_filtered.shape[0], X_coo_filtered.shape[1]))

    # Create a sparse tensor on GPU
    X_gpu = torch.sparse_coo_tensor(indices, values, dense_shape, device=device, dtype=torch.uint8)
    X_gpu = X_gpu.t()
    X_gpu = X_gpu.to(device)

    # End the timer
    block_index_end = time.time()

    # Print elapsed time for creating the block index
    print(f"BlockIndex created in: {block_index_end - block_index_start} seconds")

    # Return the final tensor
    return X_gpu

# Example usage:
# X_gpu = create_block_index('~/papers/3m/papers3m.csv', 'stopwords.txt')