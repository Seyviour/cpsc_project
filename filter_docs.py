# %%
import sys
import fasttext
import fasttext.util
import numpy as np
import heapq
import pandas as pd
from scipy.optimize import linear_sum_assignment
import random
from pprint import pprint
import os
import re
import nltk
from nltk.corpus import stopwords
import json
from functools import lru_cache
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import Levenshtein


# Load FastText embeddings
model_path = "cc.en.64.bin"
if os.path.exists(model_path):
    en_model = fasttext.load_model(model_path)
else:
    fasttext.util.download_model('en', if_exists='ignore')  # Download English model
    en_model = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(en_model, 64)
    en_model.save_model(model_path)


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


try:
    NLTK_STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("Failed to get stopwords")
    NLTK_STOPWORDS = set(stopwords.words('english'))




# %%
import time

from sentence_transformers import SentenceTransformer, util

# %%


# %%


# %%
stemmer = PorterStemmer()
# %%
SIM_VALUES_PATH = "similarity_data/sim.values.txt"
SIM_WORDS_PATH = "similarity_data/sim.words.txt"

# %%

def filter_stopwords(text):
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    words = word_tokenize(cleaned)
    filtered_words = [word for word in words if word.lower() not in NLTK_STOPWORDS]
    return filtered_words

# %%
def load_word_data(words_path=SIM_WORDS_PATH, sim_path=SIM_VALUES_PATH):
    """Load word data and similarity values."""
    words = []
    sim_values = []
    if words_path and sim_path:
        with open(words_path, "r") as f:
            words = [x.strip() for x in f.readlines()]
        with open(sim_path, "r") as f:
            sim_values = [float(x) for x in f.readlines()]
    return words, sim_values

# %%
__model = en_model
__words, __sim_values = load_word_data(SIM_WORDS_PATH, SIM_VALUES_PATH)
__num_words = len(__words)
__word2idx = {word: idx for word, idx in zip(__words, range(len(__words)))}

__hits = 0
__misses = 0

def embed_word(word):
    """Get the FastText embedding for a word."""
    return __model.get_word_vector(word)

def cosine_distance(vec1, vec2):
    """Compute cosine distance between two vectors."""
    return 1 - np.dot(vec1, vec2) / ((np.linalg.norm(vec1) * np.linalg.norm(vec2)) + 1e-8)  # numerical stability

@lru_cache(maxsize=1024)
def word_cosine_distance(word1, word2):
    """Compute cosine distance between two words."""
    word1_emb = embed_word(word1)
    word2_emb = embed_word(word2)
    return cosine_distance(word1_emb, word2_emb)
    # return int(1000 * cosine_distance(word1_emb, word2_emb))

@lru_cache(maxsize=1000)
def get_stem(word):
    return stemmer.stem(word)

@lru_cache(maxsize=2048)
def cached_word_cosine_distance(word1, word2):
    # return 0
    """Compute cached cosine distance between two words."""
    global __hits, __misses
    w1_index = __word2idx.get(word1, None)
    w2_index = __word2idx.get(word2, None)
    if w1_index and w2_index:
        __hits += 1
        cache_index = w1_index * __num_words + w2_index
        # return int(1000 * __sim_values[cache_index])
        return __sim_values[cache_index]

    else:
        __misses += 1
        # return 0
        return word_cosine_distance(word1, word2)

# %%
# len(__words) * len(__words)
len(__sim_values)

# %%
def clean_text(text):
    """
    Removes stop words and punctuation from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The cleaned text.
    """
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Get the stop words and punctuation
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    # Filter out stop words and punctuation
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in punctuation]
    
    # Join the tokens back into a string
    return ' '.join(filtered_tokens)

# %%
import zipfile

def read_file_from_zip(zip_path, file_name):
    """
    Reads the content of a specific file from a zipped folder without extracting the entire folder.

    Args:
        zip_path (str): Path to the zip file.
        file_name (str): Name of the file to read within the zip.

    Returns:
        str: Content of the file as a string.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open(file_name) as file:
            return file.read().decode('utf-8')

# %%
DEV_KNOWLEDGE_STORE = "baseline/AVeriTeC/data_store/knowledge_store/dev_knowledge_store.zip"
TRAIN_KNOWLEDGE_STORE_999 = "baseline/AVeriTeC/data_store/knowledge_store/train/train_0_999.zip"
TRAIN_KNOWLEDGE_STORE_1999 = "baseline/AVeriTeC/data_store/knowledge_store/train/train_1000_1999.zip"
TRAIN_KNOWLEDGE_STORE_3067 = "baseline/AVeriTeC/data_store/knowledge_store/train/train_2000_3067.zip"
TEST_KNOWLEDGE_STORE_499 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_0_499.zip"
TEST_KNOWLEDGE_STORE_999 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_500_999.zip"
TEST_KNOWLEDGE_STORE_1499 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_1000_1499.zip"
TEST_KNOWLEDGE_STORE_1999 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_1500_1999.zip"
TEST_KNOWLEDGE_STORE_2214 = "baseline/AVeriTeC/data_store/knowledge_store/test_updated/output_test_2000_2214.zip"

# %%
def get_knowledge_store_for_claim(claim_id, div="train"):
    references = []
    try:
        if div == "train":
            if claim_id <= 999:
                train_path = TRAIN_KNOWLEDGE_STORE_999
                claim_file = read_file_from_zip(train_path, f"{claim_id}.json")
            elif claim_id <= 1999:
                train_path = TRAIN_KNOWLEDGE_STORE_1999
                claim_file = read_file_from_zip(train_path, f"{claim_id}.json")
            else:
                train_path = TRAIN_KNOWLEDGE_STORE_3067
                claim_file = read_file_from_zip(train_path, f"data_store/train/{claim_id}.json")
            # claim_file = read_file_from_zip(train_path, f"{claim_id}.json")
        elif div == "val":
            claim_file = read_file_from_zip("baseline/AVeriTeC/data_store/knowledge_store/dev_knowledge_store.zip", f"output_dev/{claim_id}.json")
        elif div == "test":
            if claim_id <= 499:
                test_path = TEST_KNOWLEDGE_STORE_499
            elif claim_id <= 999:
                test_path = TEST_KNOWLEDGE_STORE_999
            elif claim_id <= 1499:
                test_path = TEST_KNOWLEDGE_STORE_1499
            elif claim_id <= 1999:
                test_path = TEST_KNOWLEDGE_STORE_1999
            else:
                test_path = TEST_KNOWLEDGE_STORE_2214
            claim_file = read_file_from_zip(test_path, f"{claim_id}.json")
            
    except Exception as e:
        print(e)
        return None

    for line in claim_file.splitlines():
        try:
            this_ref = json.loads(line)
            references.append(this_ref)
        except Exception as e:
            # print(e)
            continue
    return references

# %%
def optimal_matching(query_embeddings, text_embeddings):
    """Find an optimal matching between query words and text words using the Hungarian algorithm."""
    cost_matrix = np.zeros((len(query_embeddings), len(text_embeddings)))
    
    for i, q_emb in enumerate(query_embeddings):
        for j, t_emb in enumerate(text_embeddings):
            cost_matrix[i, j] = cosine_distance(q_emb, t_emb)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_score = -cost_matrix[row_ind, col_ind].sum()
    
    return total_score


cost_matrix = None #avoid having to assign memory with each call. Will resize cost matrix when shape changes
def word_optimal_matching(query_words, text_words):
    # cost_matrix = np.zeros((len(query_words), len(text_words)))
    # return 0.1
    global cost_matrix
    shape = (len(query_words), len(text_words))
    if cost_matrix is None or cost_matrix.shape != shape:
        cost_matrix = np.zeros(shape)
    else:
        cost_matrix[:] = 1
    for i, word1 in enumerate(query_words):
        for j, word2 in enumerate(text_words):
            dist = cached_word_cosine_distance(word1, word2)
            # if dist > 0.7:
            #     dist = 1
            cost_matrix[i, j] = dist
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_score = -cost_matrix[row_ind, col_ind].sum()
    lev_score = Levenshtein.ratio(col_ind, sorted(col_ind))
    # match_width = max(col_ind) - min(col_ind)
    
    
    return total_score, lev_score#, lev_score, match_width

def find_top_n_matches(query_text, target_text, patch_size=50, overlap=25, top_n=5, ret_string=False):
    """Find the top N scoring spans in the target text using a sliding window without redundant embeddings."""
    # q_words = target_words.split()
    t_words = re.sub(r'[^\w\s]', '', target_text.lower())
    target_words = t_words = word_tokenize(t_words)
    query_words = [get_stem(word) for word in filter_stopwords(query_text)]
    # target_words = [word for word in target_text.split()]
    # query_words = [word for word in query_text.split()][:500]
    low_lev_scores = []
    mid_lev_scores = []
    high_lev_scores = []

    for start in range(0, max(1, len(target_words) - patch_size + 1), patch_size - overlap):
        end = start + patch_size
        curr_target_words = [get_stem(word) for word in target_words[start:end] if word not in NLTK_STOPWORDS]
        score, lev_score = word_optimal_matching(query_words, curr_target_words)
        
        curr_target_text = " ".join(target_words[start:end])
        if ret_string:
            ret_info = curr_target_text
        else:
            ret_info = (start, end)
        
        if lev_score < 0.3:
            heapq.heappush(low_lev_scores, (score,ret_info))
            if len(low_lev_scores) > top_n:
                heapq.heappop(low_lev_scores)
        elif 0.3 <= lev_score <= 0.7:
            heapq.heappush(mid_lev_scores, (score,ret_info))
            if len(mid_lev_scores) > top_n:
                heapq.heappop(mid_lev_scores)
        else:
            heapq.heappush(high_lev_scores, (score,ret_info))
            if len(high_lev_scores) > top_n:
                heapq.heappop(high_lev_scores)

    return {
        "low_lev_scores": sorted(low_lev_scores, reverse=True),
        "mid_lev_scores": sorted(mid_lev_scores, reverse=True),
        "high_lev_scores": sorted(high_lev_scores, reverse=True),
    }

# %%
dev_tasks = None
with open("baseline/AVeriTeC/data/dev.json") as f:
    dev_tasks = json.load(f)
train_tasks = None
with open("baseline/AVeriTeC/data/train.json") as f:
    train_tasks = json.load(f)
test_tasks = None
with open("baseline/AVeriTeC/data/test.json") as f:
    test_tasks = json.load(f)

# %%
def get_claim_by_id(claim_list, claim_id):
    filtered = claim_list[claim_id:claim_id+1]
    return filtered[0] if filtered else None

# %%
def get_span_text_from_claim_doc(claim_doc, span): 
    span = span[-2:]
    claim_text = " ".join(claim_doc.get("url2text")).split()
    return " ".join(claim_text[span[0]: span[1]])

# %%
def filter_claim_doc(claim, claim_doc_dict, patch_size):
    if not isinstance(claim_doc_dict, dict):
        print("ERROR", claim_doc_dict)
        return {}
    claim_text = " ".join(claim_doc_dict.get("url2text"))
    # doc_url = claim_doc_dict.get("url")
    top_matches = find_top_n_matches(claim, claim_text, patch_size=patch_size, overlap=0, ret_string=True, top_n=2)
    claim_doc_dict["top_n"] = top_matches
    claim_doc_dict["url2text"] = ""
    # for i, match in enumerate(top_matches):
    #     claim_doc_dict["most_rel"].append(match)
    return claim_doc_dict 

# %%
def process_claim(claim_list, claim_id, div="train"):
    claim_id = claim_id
    claim = get_claim_by_id(claim_list, claim_id)
    claim_text = claim.get("claim", "")
    print("CLAIM:", claim_text)
    patch_size = 64
    claim_docs = get_knowledge_store_for_claim(claim_id, div=div)
    claim_docs = [
        filter_claim_doc(claim_text, doc, patch_size) for doc in claim_docs[:100]
    ]
    return claim_docs

# %%
print(f"calls: {__hits + __misses}")
print(f"hits: {__hits}, {(__hits/(__hits+__misses+1)):.2f}%")
print(f"misses: {__misses}, {(__misses/(__hits+__misses+1)):.2f}%")

# %%
# Example Usage
target_text = "This is a robust graph matching algorithm for string search. We apply it to find patterns in text efficiently."
query_text = "graph matching algorithm in bipartite graphs"

top_matches = find_top_n_matches(query_text, target_text, ret_string=True)
print(top_matches)
for group, deets in top_matches.items():
    if not deets:
        continue
    print(deets)
    for deet in deets:
        score, info = deet
        print(f"Match score: {score}, Span: {info}")

# %%
# from docx import Document

# def read_docx(file_path):
#     """
#     Reads the content of a .docx file.

#     Args:
#         file_path (str): Path to the .docx file.

#     Returns:
#         str: The content of the .docx file as a single string.
#     """
#     doc = Document(file_path)
#     content = []
#     for paragraph in doc.paragraphs:
#         if paragraph.text.strip(): content.append(paragraph.text)
#     return '\n'.join(content)

# %%

# Load the all-MiniLM-L6-v2 model
# minilm_model = SentenceTransformer('all-MiniLM-L6-v2')

# def retrieve_topk_docs(query, documents, top_k=5):
#     """
#     Retrieve the top-k documents for a given query using all-MiniLM-L6-v2 embeddings.

#     Args:
#         query (str): The query string.
#         documents (list): List of document strings.
#         top_k (int): Number of top documents to retrieve.

#     Returns:
#         list: Top-k documents sorted by similarity score.
#     """
#     # Compute embeddings for the query and documents
#     query_embedding = minilm_model.encode(query, convert_to_tensor=True)
#     doc_embeddings = minilm_model.encode(documents, convert_to_tensor=True)

#     # Compute cosine similarities
#     similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

#     # Get the top-k documents
#     top_k_indices = similarities.topk(k=top_k).indices
#     top_k_docs = [documents[idx] for idx in top_k_indices]

#     return top_k_docs

# def benchmark_retrieval_with_rag(query_texts, documents, matching_function, top_k=5):
#     """
#     Benchmarks the matching-based retrieval function against RAG retrieval.

#     Args:
#         query_texts (list): List of query strings.
#         documents (list): List of document strings.
#         matching_function (callable): Matching-based retrieval function.
#         top_k (int): Number of top results to consider.

#     Returns:
#         dict: Recall@k and time taken for both methods.
#     """
#     results = {"matching": {"recall": [], "time": []}, "rag": {"time": []}}

#     for query in query_texts:
#         # RAG retrieval
#         start_time = time.time()
#         rag_results = retrieve_topk_docs(query, documents, top_k=top_k)
#         rag_time = time.time() - start_time
#         results["rag"]["time"].append(rag_time)

#         # Matching-based retrieval
#         start_time = time.time()
#         matching_results = matching_function(query, documents, top_k=top_k)
#         matching_time = time.time() - start_time
#         results["matching"]["time"].append(matching_time)

#         # Calculate recall@k for matching retrieval
#         recall = len(set(matching_results).intersection(set(rag_results))) / len(rag_results)
#         results["matching"]["recall"].append(recall)

#     # Aggregate metrics
#     metrics = {
#         "matching": {
#             "avg_recall@k": sum(results["matching"]["recall"]) / len(results["matching"]["recall"]),
#             "avg_time": sum(results["matching"]["time"]) / len(results["matching"]["time"]),
#         },
#         "rag": {
#             "avg_time": sum(results["rag"]["time"]) / len(results["rag"]["time"]),
#         },
#     }

#     return metrics

# %%
from multiprocessing import Pool

# %%
def process_claim_wrapper(args):
    """Wrapper function to unpack arguments for process_claim."""
    claim_list, idx, div = args
    return idx, process_claim(claim_list, idx, div)

# %%
if __name__=="__main__":
    split = sys.argv[1]
    mod = int(sys.argv[2])
    mine = int(sys.argv[3])
    
    claim_docs_filtered = []
    
    split = split.lower().strip()
    
    if split == "val":
        tasks = dev_tasks
    elif split == "train":
        tasks = train_tasks
    elif split == "test":
        tasks = test_tasks
    else:
        raise ValueError("Invalid split value. Must be 'val', 'train', or 'test'.")
    
    
    print(f"Processing split: {split}, mod: {mod}, mine: {mine}")
        
        # Create a directory for the split if it doesn't exist
    os.makedirs(f"results/{split}", exist_ok=True)
    
    df = pd.DataFrame()
    for idx, claim in enumerate(tasks[:]):
        if idx % mod != mine:
            continue
        try:
            print(f"Processing claim with ID {idx}...")
            _claim_text = claim.get("claim", "")
            _label = claim.get("label", "")
            processed_claim = {
                "filtered_docs": process_claim(tasks, idx, split),
                "claim_id": idx,
                "claim": _claim_text,
                "label": _label
            }
            claim_docs_filtered.append(processed_claim)
                        
            # Append the new processed claim to the DataFrame
            df = pd.concat([df, pd.DataFrame([processed_claim])], ignore_index=True)
            # Write the updated DataFrame back to the file
            df.to_pickle(f"results/{split}/{split}_claims_processed_mod_{mod}_mine_{mine}.pkl")
            print(f"Successfully processed claim with ID {idx}.")

        except Exception as e:
            print(f"Failed to process claim with ID {idx}. Error: {e}")
            continue
