# %%
# This script is designed to process claims and their associated documents to find the most relevant text spans within those documents that support or refute the claim.
# It utilizes various NLP techniques and libraries for text processing, embedding, and similarity matching.
# Key functionalities include:
# - Loading and preprocessing claims and documents.
# - Generating word embeddings using FastText.
# - Filtering stopwords and performing stemming using NLTK.
# - Calculating similarity scores between claim and document text spans.
# - Employing the Hungarian algorithm for optimal matching between word sets.
# - Storing and retrieving processed claim data.

import sys
import fasttext # Library for efficient learning of word representations and sentence classification.
import fasttext.util # Utilities for FastText, such as model download and reduction.
import numpy as np # Library for numerical operations, especially array manipulation.
import heapq # Library for heap queue algorithm (priority queue). Used here to find top N matches.
import pandas as pd # Library for data manipulation and analysis, particularly for working with DataFrames.
from scipy.optimize import linear_sum_assignment # Used for the Hungarian algorithm to solve the assignment problem, finding optimal matching between word sets.
import random # Library for generating random numbers.
from pprint import pprint # For pretty-printing Python data structures.
import os # Provides a way of using operating system dependent functionality like reading or writing to the file system.
import re # Regular expression operations.
import nltk # Natural Language Toolkit, a suite of libraries and programs for symbolic and statistical natural language processing.
from nltk.corpus import stopwords # NLTK corpus of stopwords (common words like 'the', 'is', 'in').
import json # Library for working with JSON data.
from functools import lru_cache # Decorator for memoizing function calls (caching results of expensive function calls).
from nltk.stem import PorterStemmer # NLTK stemmer for reducing words to their root form.
# nltk is already imported. # import nltk
# stopwords from nltk.corpus is already imported. # from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize # NLTK tokenizer for splitting text into words.
import string # Contains a collection of string constants (e.g., punctuation).
import Levenshtein # Library for calculating Levenshtein distance (edit distance) between strings.


MAX_NUM_DOCS_PER_CLAIM = 100 # Maximum number of documents to process per claim.
MAX_NUM_TASKS = -1 # Maximum number of tasks (claims) to process. -1 means all tasks.
MAX_NUM_REPHRASALS=2 # Maximum number of rephrased versions of a claim to consider.

# Load FastText embeddings
# FastText is used to generate word embeddings, which are numerical representations of words capturing semantic meaning.
# These embeddings are crucial for calculating similarity between words and text spans.
model_path = "cc.en.64.bin" # Path to the pre-trained FastText model (reduced to 64 dimensions).
if os.path.exists(model_path):
    en_model = fasttext.load_model(model_path)
else:
    # Download and reduce the FastText model if it doesn't exist locally.
    # Reduction to 64 dimensions is done to save memory and computation time compared to the full 300-dimension model.
    print(f"FastText model not found at {model_path}. Downloading and reducing...")
    fasttext.util.download_model('en', if_exists='ignore')  # Download English model (e.g., cc.en.300.bin)
    en_model = fasttext.load_model('cc.en.300.bin') # Load the full model
    fasttext.util.reduce_model(en_model, 64) # Reduce the model dimensionality to 64.
    en_model.save_model(model_path) # Save the reduced model for future use.
    print(f"Reduced model saved to {model_path}.")


# Download NLTK resources (punkt tokenizer and stopwords corpus) if not already present.
# NLTK is used for text preprocessing tasks like tokenization (splitting text into words) and stopword removal.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt') # Corrected from 'punkt_tab' to 'punkt'
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' corpus not found. Downloading...")
    nltk.download('stopwords')


# Load NLTK stopwords.
# NLTK_STOPWORDS is a set of common English words that are often removed from text during preprocessing
# as they typically do not carry significant semantic meaning for similarity analysis.
try:
    NLTK_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    # This fallback might be redundant if the download above is successful and NLTK is properly configured.
    print("Failed to get stopwords via nltk.corpus.words('english') after attempting download. This might indicate an NLTK setup issue.")
    # As a last resort, provide a small, default list of stopwords, though this is not ideal.
    NLTK_STOPWORDS = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}




# %%
import time # Library for time-related functions (e.g., measuring execution time, though not explicitly used here).

from sentence_transformers import SentenceTransformer, util # Library for state-of-the-art sentence, text and image embeddings.
# `SentenceTransformer` can create embeddings for entire sentences or paragraphs.
# `util` provides utility functions, e.g., for calculating cosine similarity between embeddings.
# Note: While imported, SentenceTransformer objects don't seem to be explicitly used in the provided code.
# The embedding is handled by FastText at the word level. This import might be for other experiments or a remnant.

# %%


# %%


# %%
stemmer = PorterStemmer() # Initialize PorterStemmer for reducing English words to their non-inflected root form (stemming).
# %%
# Paths for precomputed similarity data. This is a caching mechanism.
# If these files exist, precomputed word similarity scores are loaded to speed up calculations
# for word pairs present in this cache, avoiding re-computation using FastText.
SIM_VALUES_PATH = "similarity_data/sim.values.txt" # File containing a flat list of similarity values.
SIM_WORDS_PATH = "similarity_data/sim.words.txt" # File containing the corresponding list of words.

# %%

def filter_stopwords(text):
    """
    Removes punctuation from text, converts it to lowercase, tokenizes it, and filters out stopwords.

    Args:
        text (str): The input text.

    Returns:
        list[str]: A list of filtered words (tokens).
    """
    cleaned = re.sub(r'[^\w\s]', '', text.lower()) # Remove punctuation (anything not a word character or whitespace) and convert to lowercase.
    words = word_tokenize(cleaned) # Tokenize the cleaned text into words using NLTK's tokenizer.
    # Filter out stopwords from the tokenized list.
    filtered_words = [word for word in words if word.lower() not in NLTK_STOPWORDS]
    return filtered_words

# %%
def load_word_data(words_path=SIM_WORDS_PATH, sim_path=SIM_VALUES_PATH):
    """
    Load precomputed word list and their pairwise similarity values from specified files.
    This data is used by `cached_word_cosine_distance` to quickly retrieve similarity scores.

    Args:
        words_path (str): Path to the file containing the list of words.
        sim_path (str): Path to the file containing the list of similarity values.
                        The similarity values are expected to correspond to a flattened similarity matrix
                        for the words in `words_path`.

    Returns:
        tuple[list[str], list[float]]: A tuple containing:
            - words (list[str]): The list of words loaded from `words_path`.
            - sim_values (list[float]): The list of similarity values loaded from `sim_path`.
            Returns empty lists if files are not found or paths are not provided.
    """
    words = []
    sim_values = []
    # Check if paths are provided and corresponding files exist.
    if words_path and sim_path and os.path.exists(words_path) and os.path.exists(sim_path):
        with open(words_path, "r") as f:
            words = [x.strip() for x in f.readlines()] # Read words, stripping whitespace.
        with open(sim_path, "r") as f:
            sim_values = [float(x) for x in f.readlines()] # Read similarity values, converting to float.
    else:
        print(f"Warning: Precomputed similarity data not loaded. Files not found: {words_path}, {sim_path}")
    return words, sim_values

# %%
# Global variables for word embeddings and precomputed similarities.
__model = en_model # The loaded FastText model (global reference).
__words, __sim_values = load_word_data(SIM_WORDS_PATH, SIM_VALUES_PATH) # Load precomputed word data.
__num_words = len(__words) # Number of words in the precomputed vocabulary. Used for indexing into `__sim_values`.
# Create a mapping from word to its index in the precomputed vocabulary for quick lookups.
__word2idx = {word: idx for word, idx in zip(__words, range(len(__words)))}

# Counters for cache hits and misses for `cached_word_cosine_distance`.
# These help in assessing the effectiveness of the precomputed similarity cache.
__hits = 0  # Incremented when a word pair's similarity is found in the precomputed cache.
__misses = 0 # Incremented when a word pair's similarity must be computed using FastText.

def embed_word(word):
    """Get the FastText word embedding (vector representation) for a given word."""
    return __model.get_word_vector(word)

def cosine_distance(vec1, vec2):
    """
    Compute cosine distance between two vectors.
    Cosine distance is defined as 1 - cosine similarity.
    A small epsilon (1e-8) is added to the denominator for numerical stability, preventing division by zero
    if one or both vectors have zero magnitude (though FastText vectors are typically non-zero).
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return 1 - dot_product / ((norm_vec1 * norm_vec2) + 1e-8)

@lru_cache(maxsize=1024) # Decorator to cache the results of this function.
                         # If `word_cosine_distance` is called again with the same `word1` and `word2`,
                         # the cached result is returned instead of recomputing.
def word_cosine_distance(word1, word2):
    """
    Compute cosine distance between the FastText embeddings of two words.
    This function is cached using `lru_cache` to avoid redundant computations for identical word pairs.
    """
    word1_emb = embed_word(word1) # Get FastText embedding for the first word.
    word2_emb = embed_word(word2) # Get FastText embedding for the second word.
    return cosine_distance(word1_emb, word2_emb) # Compute cosine distance between embeddings.
    # return int(1000 * cosine_distance(word1_emb, word2_emb)) # Alternative scaled version (commented out).

@lru_cache(maxsize=1000) # Cache results for stemming to speed up repeated stemming of the same word.
def get_stem(word):
    """Get the stem of a word using the globally initialized PorterStemmer."""
    return stemmer.stem(word)

@lru_cache(maxsize=2048) # Cache results for this potentially expensive distance calculation.
def cached_word_cosine_distance(word1, word2):
    """
    Compute cosine distance between two words, prioritizing precomputed similarity values if available.
    If both words are found in the precomputed vocabulary (`__words`), their cached similarity is retrieved.
    Otherwise, the distance is calculated on-the-fly using `word_cosine_distance` (which leverages FastText embeddings).
    This function updates global cache hit/miss counters.

    Args:
        word1 (str): The first word.
        word2 (str): The second word.

    Returns:
        float: The cosine distance between word1 and word2.
    """
    # return 0 # Placeholder for debugging (commented out).
    global __hits, __misses # Allow modification of global hit/miss counters.
    
    # Attempt to find indices of word1 and word2 in the precomputed vocabulary.
    w1_index = __word2idx.get(word1, None)
    w2_index = __word2idx.get(word2, None)
    
    # If both words are in the precomputed vocabulary and __sim_values is loaded.
    if w1_index is not None and w2_index is not None and __sim_values:
        __hits += 1
        # Calculate the index in the flattened similarity matrix `__sim_values`.
        # The matrix is stored such that sim(word_i, word_j) is at index i * num_words + j.
        # Ensure indices are within bounds.
        if 0 <= w1_index < __num_words and 0 <= w2_index < __num_words:
            cache_index = w1_index * __num_words + w2_index
            if 0 <= cache_index < len(__sim_values):
                 return __sim_values[cache_index] # Return precomputed similarity.
            else:
                # This case should ideally not happen if data is consistent.
                # Fall through to re-computation if index is out of bounds.
                print(f"Warning: Calculated cache_index {cache_index} out of bounds for __sim_values (len {len(__sim_values)}).")
                pass # Fall through to miss case
        else:
            print(f"Warning: Word indices {w1_index}, {w2_index} out of bounds for __num_words ({__num_words}).")
            pass # Fall through to miss case


    # If not found in cache or cache is problematic, compute using FastText.
    __misses += 1
    # return 0 # Placeholder for debugging (commented out).
    return word_cosine_distance(word1, word2) # Calculate distance using FastText embeddings.

# %%
# This cell, when uncommented in a Jupyter notebook, would display the length of the `__sim_values` list.
# It's useful for verifying that the precomputed similarity data has been loaded correctly.
# The length should ideally be `__num_words * __num_words`.
# len(__sim_values)

# %%
def clean_text(text):
    """
    Cleans the input text by tokenizing, removing stop words, and removing punctuation.
    The cleaned tokens are then joined back into a single string.

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned text string.
    """
    # Tokenize the text using NLTK's word_tokenize.
    tokens = word_tokenize(text)
    
    # Use the globally defined NLTK_STOPWORDS set.
    # `string.punctuation` provides a string of all punctuation characters.
    punctuation_set = set(string.punctuation)
    
    # Filter out tokens that are stopwords (case-insensitive) or punctuation.
    filtered_tokens = [
        word for word in tokens 
        if word.lower() not in NLTK_STOPWORDS and word not in punctuation_set
    ]
    
    # Join the filtered tokens back into a single string, separated by spaces.
    return ' '.join(filtered_tokens)

# %%
import zipfile # Library for working with ZIP archives.

def read_file_from_zip(zip_path, file_name):
    """
    Reads the content of a specific file from a ZIP archive without extracting the entire archive.
    This is efficient for accessing individual files within large compressed datasets.

    Args:
        zip_path (str): Path to the ZIP file.
        file_name (str): The name (including any internal path) of the file to read from the ZIP archive.

    Returns:
        str: The content of the file as a UTF-8 decoded string.
             Returns None if an error occurs during file reading (e.g., file not found in zip, bad zip file).
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref: # Open the ZIP file in read mode.
            with zip_ref.open(file_name) as file: # Open the specific file within the ZIP.
                return file.read().decode('utf-8') # Read and decode the file content.
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found in zip '{zip_path}'.")
        return None
    except Exception as e:
        print(f"Error reading file '{file_name}' from zip '{zip_path}': {e}")
        return None

# %%
# Paths to different knowledge store ZIP files for various datasets (dev, train, test splits).
# These ZIP files are expected to contain JSON documents, where each JSON file (e.g., "{claim_id}.json")
# itself contains multiple JSON objects (one per line), each representing a piece of evidence or a document.
# The naming convention suggests they are organized by dataset split and ranges of claim IDs.
DEV_KNOWLEDGE_STORE = "baseline/AVeriTeC/data_store/knowledge_store/dev_knowledge_store.zip"
TRAIN_KNOWLEDGE_STORE_999 = "baseline/AVeriTeC/data_store/knowledge_store/train/train_0_999.zip" # For claim IDs 0-999
TRAIN_KNOWLEDGE_STORE_1999 = "baseline/AVeriTeC/data_store/knowledge_store/train/train_1000_1999.zip" # For claim IDs 1000-1999
TRAIN_KNOWLEDGE_STORE_3067 = "baseline/AVeriTeC/data_store/knowledge_store/train/train_2000_3067.zip" # For claim IDs 2000-3067
TEST_KNOWLEDGE_STORE_499 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_0_499.zip" # For test claim IDs 0-499
TEST_KNOWLEDGE_STORE_999 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_500_999.zip" # For test claim IDs 500-999
TEST_KNOWLEDGE_STORE_1499 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_1000_1499.zip" # For test claim IDs 1000-1499
TEST_KNOWLEDGE_STORE_1999 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_1500_1999.zip" # For test claim IDs 1500-1999
TEST_KNOWLEDGE_STORE_2214 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_2000_2214.zip" # For test claim IDs 2000-2214

# %%
def get_knowledge_store_for_claim(claim_id, div="train"):
    """
    Retrieves the knowledge store (a list of document reference dictionaries) for a given claim ID and dataset division.
    It identifies the correct ZIP archive based on `div` and `claim_id`, then reads the corresponding JSON file
    (e.g., "{claim_id}.json") from that archive. Each line in this JSON file is parsed as a separate JSON object.

    Args:
        claim_id (int): The ID of the claim for which to retrieve documents.
        div (str): The dataset division, one of 'train', 'val' (for development), or 'test'.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a document reference
                     (e.g., containing URL, text snippets).
                     Returns None if the specified claim's data file cannot be found or read, or if parsing fails.
    """
    references = [] # List to store parsed document/reference JSON objects.
    claim_file_content = None # To store the content read from the zip file.
    
    try:
        # Determine the correct ZIP file path and the name of the file within the ZIP.
        # This logic maps a claim_id and division to a specific archive and internal file path.
        if div == "train":
            if claim_id <= 999:
                zip_path = TRAIN_KNOWLEDGE_STORE_999
                file_name_in_zip = f"{claim_id}.json"
            elif claim_id <= 1999:
                zip_path = TRAIN_KNOWLEDGE_STORE_1999
                file_name_in_zip = f"{claim_id}.json"
            else: # claim_id > 1999 for train
                zip_path = TRAIN_KNOWLEDGE_STORE_3067
                # Note: The internal path structure seems different for this specific training data segment.
                file_name_in_zip = f"data_store/train/{claim_id}.json"
            claim_file_content = read_file_from_zip(zip_path, file_name_in_zip)
        elif div == "val": # 'val' is used for the development set.
            zip_path = DEV_KNOWLEDGE_STORE
            file_name_in_zip = f"output_dev/{claim_id}.json" # Internal path for dev data.
            claim_file_content = read_file_from_zip(zip_path, file_name_in_zip)
        elif div == "test":
            # Similar logic for test set, mapping claim_id ranges to specific ZIP files.
            if claim_id <= 499: zip_path = TEST_KNOWLEDGE_STORE_499
            elif claim_id <= 999: zip_path = TEST_KNOWLEDGE_STORE_999
            elif claim_id <= 1499: zip_path = TEST_KNOWLEDGE_STORE_1499
            elif claim_id <= 1999: zip_path = TEST_KNOWLEDGE_STORE_1999
            else: zip_path = TEST_KNOWLEDGE_STORE_2214 # For claim_ids > 1999 in test
            file_name_in_zip = f"{claim_id}.json" # Assuming consistent naming within test zips.
            claim_file_content = read_file_from_zip(zip_path, file_name_in_zip)
        else:
            print(f"Error: Invalid division '{div}' specified.")
            return None
            
    except Exception as e: # Catch any unexpected errors during path determination or initial read.
        print(f"Error accessing knowledge store for claim {claim_id} in division '{div}': {e}")
        return None

    if claim_file_content is None: # If read_file_from_zip returned None (e.g., file not found).
        print(f"No content found for claim {claim_id} in division '{div}'.")
        return None

    # Process each line of the retrieved file content as a separate JSON object.
    for line_number, line in enumerate(claim_file_content.splitlines()):
        try:
            this_ref = json.loads(line) # Parse the JSON string from the current line.
            references.append(this_ref)
        except json.JSONDecodeError as e:
            # If a line is not valid JSON, print an error and skip it.
            print(f"Skipping invalid JSON line {line_number + 1} for claim {claim_id} in '{div}': {e}")
            continue
    return references

# %%
def optimal_matching(query_embeddings, text_embeddings):
    """
    Finds an optimal one-to-one matching between query word embeddings and text word embeddings
    using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    The "cost" of matching two embeddings is their cosine distance. The algorithm minimizes the total cost.

    Args:
        query_embeddings (list[np.array]): A list of numpy arrays, where each array is the embedding for a query word.
        text_embeddings (list[np.array]): A list of numpy arrays, where each array is the embedding for a text word.

    Returns:
        float: The total score of the optimal matching. This score is the negative sum of the
               cosine distances of the matched pairs (so a higher, i.e., less negative, score is better).
               Returns 0.0 if either query_embeddings or text_embeddings is empty.
    """
    if not query_embeddings or not text_embeddings:
        return 0.0 # Cannot perform matching if one list is empty.

    # Initialize a cost matrix where rows correspond to query embeddings and columns to text embeddings.
    cost_matrix = np.zeros((len(query_embeddings), len(text_embeddings)))
    
    # Populate the cost matrix with cosine distances. cost_matrix[i, j] = distance(query_i, text_j).
    for i, q_emb in enumerate(query_embeddings):
        for j, t_emb in enumerate(text_embeddings):
            cost_matrix[i, j] = cosine_distance(q_emb, t_emb)
    
    # Use linear_sum_assignment (Hungarian algorithm) to find the optimal assignment
    # that minimizes the sum of costs (distances).
    # `row_ind[k]` is matched with `col_ind[k]`.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Calculate the total score: sum of costs (distances) for the optimal assignment.
    # We negate the sum because we want to maximize similarity (which is equivalent to minimizing distance).
    # A higher score (closer to 0 or positive, if similarity was used) indicates a better match.
    total_score = -cost_matrix[row_ind, col_ind].sum()
    
    return total_score


# Global cost matrix to potentially reuse memory in `word_optimal_matching`.
# This is an optimization to avoid reallocating the matrix on every call if dimensions are often similar.
# However, care must be taken if using this in multithreaded contexts (not an issue here as it's used with multiprocessing Pool later).
cost_matrix = None 
def word_optimal_matching(query_words, text_words):
    """
    Calculates an optimal matching score between a list of query words and a list of text words.
    1. It computes a cost matrix where entry (i,j) is the `cached_word_cosine_distance`
       between query_words[i] and text_words[j].
    2. It uses the Hungarian algorithm (`linear_sum_assignment`) to find the assignment of query
       words to text words that minimizes the total distance.
    3. It also calculates a Levenshtein ratio score based on the order of matched indices in the text,
       which can indicate the contiguity or sequential coherence of the matched words.

    Args:
        query_words (list[str]): A list of preprocessed (e.g., stemmed, filtered) words from the query.
        text_words (list[str]): A list of preprocessed words from the text span.

    Returns:
        tuple[float, float]: A tuple containing:
            - total_score (float): The negative sum of cosine distances for the optimal matching. Higher is better.
                                   Returns 0.0 if either word list is empty.
            - lev_score (float): Levenshtein ratio comparing the sequence of matched text word indices
                                 (col_ind) to its sorted version. A score closer to 1.0 suggests the matched
                                 words in the text span appear in a more contiguous or ordered manner.
                                 Returns 0.0 if no matches are made.
    """
    if not query_words or not text_words:
        return 0.0, 0.0 # No matching possible if one list is empty.

    global cost_matrix # Use the globally defined cost_matrix for potential memory reuse.
    shape = (len(query_words), len(text_words))

    # Resize the global cost matrix if it's not initialized or its shape is different from the current need.
    if cost_matrix is None or cost_matrix.shape != shape:
        cost_matrix = np.zeros(shape)
    else:
        # If reusing an existing matrix, fill it with a default value (e.g., 1, representing max distance for cosine).
        # This ensures values from previous computations don't interfere.
        cost_matrix[:] = 1.0 # Initialize with max distance

    # Populate the cost matrix using `cached_word_cosine_distance`.
    for i, word1 in enumerate(query_words):
        for j, word2 in enumerate(text_words):
            dist = cached_word_cosine_distance(word1, word2)
            # A commented-out option to threshold distances:
            # if dist > 0.7:
            #     dist = 1
            cost_matrix[i, j] = dist
            
    # Apply the Hungarian algorithm to find the optimal assignment.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # Calculate total_score: negative sum of distances of the matched pairs.
    total_score = -cost_matrix[row_ind, col_ind].sum()
    
    # Calculate Levenshtein ratio on the column (text word) indices of the matches.
    # This measures the "contiguity" or "orderliness" of the matched words in the text.
    # Convert indices to strings for Levenshtein.ratio, which expects strings.
    # If col_ind is empty (no matches), Levenshtein ratio is 0.
    if len(col_ind) > 0:
        # Create string representations of the index sequences.
        # Using a separator to distinguish multi-digit numbers, e.g., "1-2-10" vs "12-10".
        # Simpler approach: Levenshtein.ratio works on sequences directly, not just strings.
        # However, the original code was `Levenshtein.ratio(col_ind, sorted(col_ind))`.
        # This is problematic as `Levenshtein.ratio` expects strings or sequences of same-type items.
        # np.array of ints is a valid sequence.
        lev_score = Levenshtein.ratio(col_ind, np.sort(col_ind))
    else:
        lev_score = 0.0
        
    # A commented-out alternative metric:
    # match_width = max(col_ind) - min(col_ind) if len(col_ind) > 0 else 0
    
    return total_score, lev_score

def find_top_n_matches(query_text_options, target_text, patch_size=50, overlap=25, top_n=5, ret_string=False):
    """
    Finds the top N scoring text spans (patches) in `target_text` that match any of the `query_text_options`.
    It uses a sliding window to create patches from `target_text`. For each patch, it calculates an
    optimal matching score against each query option using `word_optimal_matching`.
    The scores are categorized based on the Levenshtein scores (contiguity of matched words)
    into 'low', 'mid', and 'high' Levenshtein score heaps.

    Args:
        query_text_options (list[str]): A list of query strings (e.g., original claim and its rephrasals).
        target_text (str): The text to search within (e.g., document content).
        patch_size (int): The number of words in each sliding window patch from `target_text`.
        overlap (int): The number of words overlapping between consecutive patches.
        top_n (int): The number of top matches to return for each Levenshtein score category.
        ret_string (bool): If True, return the matched text span string; otherwise, return (start, end) word indices.

    Returns:
        dict: A dictionary with keys "low_lev_scores", "mid_lev_scores", "high_lev_scores".
              Each key maps to a list of top N matches for that category, sorted by match score (descending).
              Each match is a tuple (score, info), where `score` is from `word_optimal_matching`
              and `info` is either the text span string or (start_index, end_index) tuple.
    """
    # Preprocess target text: remove punctuation, lowercase, tokenize.
    # `t_words` here seems to be a typo and should be `target_words_cleaned_tokenized` or similar for clarity.
    # The original code `target_words = t_words = word_tokenize(t_words)` is a bit redundant.
    cleaned_target_text = re.sub(r'[^\w\s]', '', target_text.lower())
    tokenized_target_words = word_tokenize(cleaned_target_text)

    # Preprocess query options: for each option, filter stopwords and get stems.
    # This results in a list of lists, where each inner list contains processed words of a query option.
    processed_query_options = [[get_stem(word) for word in filter_stopwords(option)] for option in query_text_options]
    # Filter out any query options that became empty after processing.
    processed_query_options = [opt for opt in processed_query_options if opt]

    if not processed_query_options: # If all query options are empty after processing
        return {"low_lev_scores": [], "mid_lev_scores": [], "high_lev_scores": []}

    # Heaps to store top N scores for different Levenshtein score categories.
    # heapq implements a min-heap. To keep the N largest scores, we can push (score, item)
    # and if len(heap) > top_n, call heapq.heappop() to remove the smallest.
    # Or, use heapq.heappushpop if the new item might replace the smallest.
    low_lev_scores = []  # For Levenshtein scores < 0.3
    mid_lev_scores = []  # For Levenshtein scores between 0.3 and 0.7 (inclusive)
    high_lev_scores = [] # For Levenshtein scores > 0.7

    # Sliding window over the tokenized target words.
    # The step is (patch_size - overlap) to create overlapping windows.
    # `max(1, ...)` ensures at least one iteration if target_words is shorter than patch_size.
    # `len(tokenized_target_words) - patch_size + 1` is the number of possible start positions for a full patch.
    for start in range(0, max(1, len(tokenized_target_words) - patch_size + 1), patch_size - overlap):
        end = start + patch_size
        # Current patch of target words: get stems and filter stopwords.
        current_patch_target_words = [get_stem(word) for word in tokenized_target_words[start:end] if word not in NLTK_STOPWORDS]
        
        if not current_patch_target_words: # Skip if the current patch becomes empty after processing.
            continue

        # Calculate matching scores for the current patch against all processed query options.
        matchings_for_patch = [word_optimal_matching(option_words, current_patch_target_words) for option_words in processed_query_options]
        
        if not matchings_for_patch: # If no valid matchings were found for this patch.
            continue

        # Select the best score (highest `total_score`) among all query options for the current patch.
        # `word_optimal_matching` returns (total_score, lev_score). We maximize by `total_score`.
        best_score, best_lev_score = max(matchings_for_patch, key = lambda x: x[0])
        
        # Determine what information to store for this match (text span string or (start, end) indices).
        # The indices are relative to the `tokenized_target_words` list.
        actual_text_span = " ".join(tokenized_target_words[start:end])
        return_info = actual_text_span if ret_string else (start, end)
        
        # Categorize and store the score in the appropriate heap, maintaining top_n highest scores.
        # This uses heappush and heappop to maintain the N largest elements in a min-heap.
        # When heap size exceeds N, the smallest element is popped.
        # A more direct way for N largest is to push (-score, item) and pop if heap too large, then negate scores at the end.
        # Or, as done here: push (score, item), and if heap grows too large, pop smallest.
        # This works correctly because sorted(..., reverse=True) is used at the end.

        item_to_add = (best_score, return_info)
        if best_lev_score < 0.3:
            heapq.heappush(low_lev_scores, item_to_add)
            if len(low_lev_scores) > top_n:
                heapq.heappop(low_lev_scores)
        elif 0.3 <= best_lev_score <= 0.7:
            heapq.heappush(mid_lev_scores, item_to_add)
            if len(mid_lev_scores) > top_n:
                heapq.heappop(mid_lev_scores)
        else: # best_lev_score > 0.7
            heapq.heappush(high_lev_scores, item_to_add)
            if len(high_lev_scores) > top_n:
                heapq.heappop(high_lev_scores)

    # Prepare the final results: sort the collected scores in each category by score (descending).
    # The heaps now contain the top N (or fewer) matches, but not necessarily sorted.
    ret_docs = {
        "low_lev_scores": sorted(low_lev_scores, key=lambda x: x[0], reverse=True),
        "mid_lev_scores": sorted(mid_lev_scores, key=lambda x: x[0], reverse=True),
        "high_lev_scores": sorted(high_lev_scores, key=lambda x: x[0], reverse=True),
    }
    # print(ret_docs) # For debugging
    return ret_docs

# %%
# Load task data (claims) from JSON files for different splits (dev, train, test).
# These files are expected to contain lists of claim objects, each with details like "claim_id", "claim" text, "label", etc.
# These lists are loaded into global variables `dev_tasks`, `train_tasks`, `test_tasks`.
dev_tasks = None
try:
    with open("baseline/AVeriTeC/data/dev.json") as f:
        dev_tasks = json.load(f) # List of development set tasks (claims).
except FileNotFoundError:
    print("Warning: Development tasks file 'baseline/AVeriTeC/data/dev.json' not found.")
except json.JSONDecodeError:
    print("Error: Could not decode JSON from 'baseline/AVeriTeC/data/dev.json'.")

train_tasks = None
try:
    with open("baseline/AVeriTeC/data/train.json") as f:
        train_tasks = json.load(f) # List of training set tasks.
except FileNotFoundError:
    print("Warning: Training tasks file 'baseline/AVeriTeC/data/train.json' not found.")
except json.JSONDecodeError:
    print("Error: Could not decode JSON from 'baseline/AVeriTeC/data/train.json'.")

test_tasks = None
try:
    with open("baseline/AVeriTeC/data/test.json") as f:
        test_tasks = json.load(f) # List of test set tasks.
except FileNotFoundError:
    print("Warning: Test tasks file 'baseline/AVeriTeC/data/test.json' not found.")
except json.JSONDecodeError:
    print("Error: Could not decode JSON from 'baseline/AVeriTeC/data/test.json'.")

# %%
def get_claim_by_id(claim_list, claim_id):
    """
    Retrieves a specific claim (dictionary) from a list of claims by its ID (which is its index in the list).

    Args:
        claim_list (list[dict]): The list of claims (e.g., `train_tasks`, `dev_tasks`).
        claim_id (int): The ID (index) of the claim to retrieve.

    Returns:
        dict: The claim dictionary if the `claim_id` is a valid index within `claim_list`.
              Otherwise, returns None.
    """
    # The original slicing `claim_list[claim_id:claim_id+1]` is a way to get a single-item list
    # or an empty list if out of bounds, then taking `filtered[0] if filtered`.
    # A more direct and Pythonic way to check index validity:
    if claim_list is not None and 0 <= claim_id < len(claim_list):
        return claim_list[claim_id]
    return None # Return None if claim_list is None or index is out of bounds.

# %%
def get_span_text_from_claim_doc(claim_doc, span): 
    """
    Extracts a text span from a claim document's text based on start and end token indices.
    The `claim_doc` is expected to have a "url2text" field, which might be a list of strings (sentences/paragraphs)
    or a single string. These are joined, then split into words to honor token-based `span` indices.

    Args:
        claim_doc (dict): The claim document dictionary. Must contain a "url2text" field,
                          which should be a string or a list of strings.
        span (list or tuple): A sequence where the last two elements are interpreted as the
                              start and end token indices for the desired span.

    Returns:
        str: The extracted text span (words joined by spaces).
             Returns an empty string if "url2text" is missing/malformed, span indices are invalid, or any error occurs.
    """
    if not (isinstance(span, (list, tuple)) and len(span) >= 2):
        print(f"Error: Span {span} is not a valid list/tuple of at least two elements.")
        return ""
        
    span_indices = span[-2:] # Get the last two elements as [start_index, end_index].
    
    try:
        # Get the text content from the document.
        doc_text_content = claim_doc.get("url2text")
        if isinstance(doc_text_content, list):
            full_text = " ".join(doc_text_content)
        elif isinstance(doc_text_content, str):
            full_text = doc_text_content
        else:
            print(f"Warning: 'url2text' field in claim_doc is not a list or string (found {type(doc_text_content)}).")
            return ""
            
        claim_text_words = full_text.split() # Split the full text into a list of words.
        
        start_idx, end_idx = int(span_indices[0]), int(span_indices[1])
        
        # Ensure indices are valid.
        if not (0 <= start_idx <= end_idx <= len(claim_text_words)):
            print(f"Warning: Invalid span indices [{start_idx}, {end_idx}] for text of length {len(claim_text_words)}.")
            return ""
            
        return " ".join(claim_text_words[start_idx:end_idx]) # Extract and join the words in the span.
    except Exception as e:
        print(f"Error extracting span text for span {span_indices}: {e}")
        return ""

# %%
def filter_claim_doc(claim_options, claim_doc_dict, patch_size):
    """
    Processes a single document associated with a claim to find relevant text spans.
    It uses `find_top_n_matches` to identify text patches within the document that best match
    the provided `claim_options` (original claim text and its rephrasals).
    The 'url2text' field of the document dictionary is cleared after processing to save memory/space,
    as the relevant information is now in the 'top_n' field.

    Args:
        claim_options (list[str]): List of claim texts (original claim and any rephrasals).
        claim_doc_dict (dict): The document dictionary. Expected to have a 'url2text' field
                               (list of strings or a single string representing document content).
        patch_size (int): The size of text patches (in words) to analyze within the document.

    Returns:
        dict: The modified document dictionary. A 'top_n' field containing the results from
              `find_top_n_matches` is added. The 'url2text' field is cleared (set to "").
              Returns an empty dictionary if the input `claim_doc_dict` is not a dictionary.
    """
    if not isinstance(claim_doc_dict, dict):
        print(f"ERROR: Expected a dictionary for claim_doc_dict, but got {type(claim_doc_dict)}.")
        return {} # Return empty dict for invalid input type.
    
    # Consolidate document text from 'url2text' field.
    doc_text_list_or_str = claim_doc_dict.get("url2text", []) # Default to empty list if not found.
    if isinstance(doc_text_list_or_str, list):
        document_full_text = " ".join(doc_text_list_or_str)
    elif isinstance(doc_text_list_or_str, str):
        document_full_text = doc_text_list_or_str
    else:
        # If 'url2text' is neither list nor string, treat as empty.
        print(f"Warning: 'url2text' in doc (ID: {claim_doc_dict.get('id', 'Unknown')}) is neither list nor str. Type: {type(doc_text_list_or_str)}.")
        document_full_text = ""

    if not document_full_text.strip(): # If document text is empty or only whitespace.
        # Add empty results for 'top_n' if no text to process.
        claim_doc_dict["top_n"] = {"low_lev_scores": [], "mid_lev_scores": [], "high_lev_scores": []}
    else:
        # Find top N matching spans in the document text.
        # Overlap is set to half of patch_size for better coverage (original had 0).
        # `top_n=2` means we are looking for up to 2 best matches in each Levenshtein category.
        top_matches = find_top_n_matches(
            query_text_options=claim_options, 
            target_text=document_full_text, 
            patch_size=patch_size, 
            overlap=patch_size // 2, 
            ret_string=True, # Return the text of the span.
            top_n=2 
        )
        claim_doc_dict["top_n"] = top_matches
        
    claim_doc_dict["url2text"] = "" # Clear original text from dict to save space after processing.
    return claim_doc_dict 

# %%
def process_claim(claim_list, claim_id, div="train", claim_rephrasals=None):
    """
    Processes a single claim by:
    1. Retrieving the claim data (including its text) using `claim_id`.
    2. Fetching all associated documents (the "knowledge store") for this claim.
    3. For each document, calling `filter_claim_doc` to find and store the most relevant text spans
       that match the claim text (and its rephrasals, if provided).

    Args:
        claim_list (list[dict]): The list of all claims for the current dataset split (e.g., `train_tasks`).
        claim_id (int): The ID (index in `claim_list`) of the specific claim to process.
        div (str): The dataset division ('train', 'val', 'test') to specify which knowledge store to use.
        claim_rephrasals (list[str], optional): A list of rephrased versions of the claim text.
                                                Defaults to None if no rephrasals are available.

    Returns:
        list[dict]: A list of processed document dictionaries. Each dictionary corresponds to a document
                     from the knowledge store and includes a 'top_n' field with the identified relevant spans.
                     Returns an empty list if the claim is not found or no documents are associated with it.
    """
    # The line `claim_id = claim_id` is redundant as claim_id is already an argument.
    
    # Retrieve the specific claim data from the list of all claims.
    claim_data_obj = get_claim_by_id(claim_list, claim_id)
    if not claim_data_obj:
        print(f"Claim with ID {claim_id} not found in the provided claim list for division '{div}'.")
        return [] # Return empty list if claim data cannot be retrieved.

    claim_text_original = claim_data_obj.get("claim", "") # Get the main text of the claim.
    print(f"Processing CLAIM (ID: {claim_id}, Div: {div}): \"{claim_text_original}\"")
    
    patch_size = 64 # Define the patch size (in words) for analyzing document text.
    
    # Get all documents associated with this claim from the appropriate knowledge store.
    raw_claim_documents = get_knowledge_store_for_claim(claim_id, div=div)
    if raw_claim_documents is None: # If get_knowledge_store_for_claim returned None (e.g., error or no file).
        print(f"No knowledge store documents found for claim ID {claim_id} in division '{div}'.")
        return [] # Return empty list if no documents are found.
    if not raw_claim_documents: # If an empty list was returned.
        print(f"Knowledge store for claim ID {claim_id} in division '{div}' is empty.")
        return []


    # Prepare a list of query options: the original claim text plus any valid rephrasals.
    query_options_for_matching = [claim_text_original]
    if claim_rephrasals and isinstance(claim_rephrasals, list):
        # Add valid (non-empty string) rephrasals to the list.
        valid_rephrasals = [str(r) for r in claim_rephrasals if str(r).strip()]
        if valid_rephrasals:
            query_options_for_matching.extend(valid_rephrasals)
            # print(f"Using {len(valid_rephrasals)} rephrasals for claim {claim_id}.")
    
    # Process each document associated with the claim, up to MAX_NUM_DOCS_PER_CLAIM.
    processed_docs_list = []
    for doc_idx, doc_dict in enumerate(raw_claim_documents[:MAX_NUM_DOCS_PER_CLAIM]):
        if not isinstance(doc_dict, dict): # Ensure the document entry is a dictionary.
            print(f"Warning: Skipping invalid document entry (not a dict, index {doc_idx}) for claim {claim_id}.")
            continue
        # Filter the document to find relevant spans and add results to its dictionary.
        processed_doc = filter_claim_doc(query_options_for_matching, doc_dict, patch_size)
        processed_docs_list.append(processed_doc)
        
    return processed_docs_list

# %%
# Print cache hit/miss statistics for the `cached_word_cosine_distance` function.
# This helps in understanding the effectiveness of the precomputed word similarity cache.
# Avoid ZeroDivisionError if no calls have been made yet.
total_cached_calls = __hits + __misses
if total_cached_calls > 0:
    hit_percentage = (__hits / total_cached_calls) * 100
    miss_percentage = (__misses / total_cached_calls) * 100
    print(f"\n--- Word Cosine Distance Cache Statistics ---")
    print(f"Total calls to cached_word_cosine_distance: {total_cached_calls}")
    print(f"Cache hits (precomputed similarity used): {__hits} ({hit_percentage:.2f}%)")
    print(f"Cache misses (FastText used): {__misses} ({miss_percentage:.2f}%)")
    print(f"--- End Cache Statistics ---\n")

else:
    print("\n--- Word Cosine Distance Cache Statistics ---")
    print("No calls made to cached_word_cosine_distance yet.")
    print(f"--- End Cache Statistics ---\n")


# %%
# Example Usage of `find_top_n_matches` function.
# This block demonstrates how to use the function and what its output looks like.
# It only runs when the script is executed directly (not when imported as a module).
if __name__ == "__main__": # Check if the script is the main program being run.
    print("\n--- Example Usage of find_top_n_matches ---")
    example_target_text = "This is a robust graph matching algorithm for string search. We apply it to find patterns in text efficiently."
    example_query_text = "graph matching algorithm in bipartite graphs"

    # Call find_top_n_matches with example inputs.
    example_top_matches = find_top_n_matches(
        query_text_options=[example_query_text], 
        target_text=example_target_text, 
        ret_string=True, # Return matched spans as strings.
        top_n=2 # Find top 2 matches per category.
    )
    
    pprint(example_top_matches) # Pretty-print the entire result dictionary.
    
    # Iterate through the results and print them in a more readable format.
    for group_name, match_details_list in example_top_matches.items():
        if not match_details_list: # Skip if no matches in this category.
            continue
        print(f"\nTop matches in category: '{group_name}'")
        for detail_item in match_details_list:
            score_value, info_span_text = detail_item
            print(f"  Match score: {score_value:.4f}, Span: \"{info_span_text}\"")
    print("--- End of Example Usage ---\n")


# %%
from multiprocessing import Pool # For parallel processing using a pool of worker processes.

# %%
def process_claim_wrapper(args):
    """
    Wrapper function to unpack arguments for `process_claim` when using `multiprocessing.Pool.map`.
    `Pool.map` expects a function that takes a single argument. This wrapper takes a tuple of arguments
    and unpacks it to call `process_claim` with multiple arguments.

    Args:
        args (tuple): A tuple containing the arguments for `process_claim`.
                      Expected order: (claim_list, claim_idx, dataset_division, rephrasals_for_this_claim)

    Returns:
        tuple: A tuple containing (claim_idx, result_from_process_claim).
               This allows associating results back to their original claim index after parallel processing.
    """
    # Unpack arguments from the input tuple.
    claim_list_arg, claim_idx_arg, div_arg, rephrasals_arg = args
    # Call the main processing function with unpacked arguments.
    result = process_claim(claim_list_arg, claim_idx_arg, div_arg, claim_rephrasals=rephrasals_arg)
    return claim_idx_arg, result


# %%
# Main execution block: This code runs when the script is executed from the command line.
# It handles command-line arguments, loads data, processes claims (sequentially in this version,
# though `Pool` is imported, it's not used in the loop here for `process_claim_wrapper`), and saves results.
if __name__=="__main__":
    # Check for sufficient command-line arguments.
    if len(sys.argv) < 5:
        print("Usage: python filter_docs.py <split> <mod> <mine> <use_rephrasals_y_n>")
        print("  <split>: 'train', 'val', or 'test'")
        print("  <mod>: An integer for distributing work (e.g., 3 for 3 workers)")
        print("  <mine>: An integer for this worker's ID (e.g., 0, 1, or 2 if mod is 3)")
        print("  <use_rephrasals_y_n>: 'y' or 'n' to indicate whether to use claim rephrasals")
        print("\nExample: python filter_docs.py train 3 0 y")
        # Set default arguments for analysis or testing if not provided, to prevent script crash.
        print("\nSetting default args for analysis due to insufficient command-line args: split='val', mod=1, mine=0, use_rephrasals='n'")
        # sys.argv.extend(['val', '1', '0', 'n']) # This would modify sys.argv, can be risky.
        # Instead, assign default values directly to variables if sys.argv is too short.
        default_split, default_mod, default_mine, default_use_rephrasals = 'val', 1, 0, 'n'
        split_arg = sys.argv[1] if len(sys.argv) > 1 else default_split
        mod_arg = sys.argv[2] if len(sys.argv) > 2 else default_mod
        mine_arg = sys.argv[3] if len(sys.argv) > 3 else default_mine
        use_rephrasals_arg = sys.argv[4] if len(sys.argv) > 4 else default_use_rephrasals
    else:
        split_arg = sys.argv[1]
        mod_arg = sys.argv[2]
        mine_arg = sys.argv[3]
        use_rephrasals_arg = sys.argv[4]

    current_split = split_arg.lower().strip() # Dataset split: 'train', 'val', or 'test'.
    
    # 'mod_val' and 'mine_val' are used for distributing processing across multiple script runs (parallelization by data sharding).
    # E.g., if mod_val=3, mine_val=0 processes claims 0, 3, 6, ...
    # if mod_val=3, mine_val=1 processes claims 1, 4, 7, ...
    try:
        mod_val = int(mod_arg)
        mine_val = int(mine_arg)
    except ValueError:
        print("Error: <mod> and <mine> arguments must be integers.")
        sys.exit(1) # Exit if conversion fails.

    should_use_rephrasals_input = use_rephrasals_arg.lower().strip()
    
    # Select the appropriate task list (list of claims) based on the 'current_split' argument.
    active_tasks_list = None
    if current_split == "val":
        active_tasks_list = dev_tasks
    elif current_split == "train":
        active_tasks_list = train_tasks
    elif current_split == "test":
        active_tasks_list = test_tasks
    else:
        print(f"Error: Invalid split value '{current_split}'. Must be 'val', 'train', or 'test'.")
        sys.exit(1) # Exit for invalid split.

    if active_tasks_list is None:
        print(f"Tasks for split '{current_split}' could not be loaded (is the JSON file present and valid?). Exiting.")
        sys.exit(1) # Exit if tasks are not loaded.
    
    print(f"\nStarting processing for split: '{current_split}', mod: {mod_val}, mine: {mine_val}, use_rephrasals: '{should_use_rephrasals_input}'")
    
    # Create base output directory for results if it doesn't exist.
    base_results_dir = f"results/{current_split}"
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Load rephrased claims if requested.
    claim_rephrasals_map = {} # Dictionary to store claim rephrasals: claim_id -> list_of_rephrasal_strings.
    is_rephrasal_enabled = False
    if should_use_rephrasals_input == 'y':
        is_rephrasal_enabled = True
        # Potentially use a subdirectory for results generated with rephrasals.
        # current_results_dir = f"{base_results_dir}/rephrasals"
        # os.makedirs(current_results_dir, exist_ok=True)
        current_results_dir = base_results_dir # Or save in the same base directory, filename will distinguish.
        
        rephrasal_data_file_path = f"rephrasals/{current_split}.with_rephrasals.json"
        try:
            # Load rephrasals from a JSON file. Expected format: a list of objects,
            # each with "claim_id" and "rephrasal_list".
            rephrasal_input_df = pd.read_json(rephrasal_data_file_path)
            # Convert DataFrame to a dictionary for easy lookup by claim_id.
            claim_rephrasals_map = dict(zip(rephrasal_input_df["claim_id"], rephrasal_input_df["rephrasal_list"]))
            print(f"Successfully loaded {len(claim_rephrasals_map)} rephrasals from '{rephrasal_data_file_path}'")
        except FileNotFoundError:
            print(f"Warning: Rephrasal file not found: '{rephrasal_data_file_path}'. Proceeding without rephrasals.")
            is_rephrasal_enabled = False
        except Exception as e: # Catch other errors like JSON parsing issues or incorrect format.
            print(f"Warning: Failed to load or parse rephrasals from '{rephrasal_data_file_path}': {e}. Proceeding without rephrasals.")
            is_rephrasal_enabled = False
    else: # Rephrasals not enabled
        current_results_dir = base_results_dir

    # Initialize an empty list to store dictionaries of processed claim data, then convert to DataFrame.
    all_processed_claims_data = []
    
    # Determine which claims to process based on MAX_NUM_TASKS (global constant).
    # If MAX_NUM_TASKS is -1, all tasks are processed. Otherwise, only up to MAX_NUM_TASKS.
    tasks_for_this_run = active_tasks_list
    if MAX_NUM_TASKS != -1 and MAX_NUM_TASKS < len(active_tasks_list):
        tasks_for_this_run = active_tasks_list[:MAX_NUM_TASKS]

    # Iterate through the selected claims.
    # `idx` is the original index of the claim in `tasks_for_this_run`.
    # `claim_content_dict` is the dictionary for the current claim.
    for idx, claim_content_dict in enumerate(tasks_for_this_run):
        # Distribute work: process only if `idx % mod_val == mine_val`.
        if idx % mod_val != mine_val: 
            continue # Skip claims not assigned to this worker instance.
        
        try:
            actual_claim_id = claim_content_dict.get("claim_id", idx) # Prefer "claim_id" field, fallback to index.
            print(f"\nProcessing claim with internal ID {idx} (data claim_id: {actual_claim_id})...")
            
            claim_text_from_data = claim_content_dict.get("claim", "")
            claim_label_from_data = claim_content_dict.get("label", "") # Assuming 'label' might exist.
            
            # Get rephrasals for the current claim_id, limited by MAX_NUM_REPHRASALS.
            rephrasals_for_current_claim = []
            if is_rephrasal_enabled:
                # Use `actual_claim_id` for lookup in the rephrasals map.
                rephrasals_for_current_claim = claim_rephrasals_map.get(actual_claim_id, [])[:MAX_NUM_REPHRASALS]
            
            # print(f"DEBUG: Rephrasals for claim {actual_claim_id}: {rephrasals_for_current_claim}") # Debug print

            # Core processing step for the current claim.
            # Pass `active_tasks_list` (full list for the split) and `idx` (current index in `tasks_for_this_run`).
            filtered_docs_for_claim = process_claim(
                active_tasks_list, # The full list of tasks for the split, from which to get claim by index.
                idx,               # The current index being processed.
                current_split, 
                claim_rephrasals=rephrasals_for_current_claim
            )
            
            # Structure the processed data for this claim.
            output_claim_data_dict = {
                "filtered_docs": filtered_docs_for_claim, # List of documents with their filtered spans.
                "claim_id": actual_claim_id, # Store the original claim_id from data.
                "claim": claim_text_from_data,
                "label": claim_label_from_data
            }
            # print(f"DEBUG: Processed data for claim {actual_claim_id}: {output_claim_data_dict}") # Debug print
                        
            all_processed_claims_data.append(output_claim_data_dict) # Add to list.
            
            # Periodically save the DataFrame to a pickle file to prevent data loss.
            # This is done after each claim is processed.
            # The filename includes split, mod, and mine to distinguish outputs from different runs/workers.
            # Suffix `_TEMP` for intermediate saves.
            temp_output_filename = f"{current_results_dir}/{current_split}_claims_proc_m{mod_val}_i{mine_val}_TEMP.pkl"
            # Create DataFrame from the current list of processed claims for saving.
            current_df_to_save = pd.DataFrame(all_processed_claims_data)
            current_df_to_save.to_pickle(temp_output_filename)
            print(f"Successfully processed claim with internal ID {idx}. Intermediate results saved to '{temp_output_filename}'")

        except Exception as e:
            # Catch any exceptions during the processing of a single claim and continue with the next.
            print(f"ERROR: Failed to process claim with internal ID {idx}. Error: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging.
            continue # Move to the next claim.
    
    # After processing all assigned claims, save the final DataFrame.
    final_output_df = pd.DataFrame(all_processed_claims_data)
    # Filename for the final consolidated results from this worker.
    final_output_filename = f"{current_results_dir}/{current_split}_claims_proc_m{mod_val}_i{mine_val}_FINAL.pkl"
    final_output_df.to_pickle(final_output_filename)
    
    print(f"\nFinished processing all assigned claims for split: '{current_split}', mod: {mod_val}, mine: {mine_val}.")
    print(f"Total claims processed by this worker: {len(all_processed_claims_data)}")
    print(f"Final results saved to '{final_output_filename}'")

# %%
# Final empty cell, typical in Jupyter notebooks or scripts developed with notebook-like cell execution.
