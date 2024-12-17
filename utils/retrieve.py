import sys
import json
import nltk
# nltk.download('punkt_tab')
import torch
import argparse


import numpy as np
from copy import deepcopy
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
from typing import List, Dict, Any, Tuple
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModel, AutoTokenizer

sys.path.append('.')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('corpus')


def calculate_bm25_similarity(documents: List[str], query: str, bm25: BM25Okapi) -> List[Tuple[int, float]]:
    """
    Calculate BM25 similarity for a single query against all documents.

    :param documents: List of documents.
    :param query: A single query.
    :param bm25: An instance of BM25Okapi.
    :return: A list of tuples containing document index and similarity score.
    """
    query_tokens = word_tokenize(query.lower())
    doc_scores = bm25.get_scores(query_tokens)
    return [(doc_idx, float(score)) for doc_idx, score in enumerate(doc_scores)]

# Define a helper function for parallel processing
def process_query(query):
    return sorted(calculate_bm25_similarity(query[1], query[0], query[2]), key=lambda x: x[1], reverse=True)[:query[3]]

def bm25_similarity_multiprocess(documents: List[str], queries: List[str], top_k) -> List[List[Tuple[int, float]]]:
    """
    Calculate BM25 similarity for multiple queries against all documents using multiple processes.

    :param documents: List of documents.
    :param queries: List of queries.
    :param top_k: Number of top documents to return for each query.
    :return: A list where each element corresponds to a query and contains a list of top_k documents and their scores.
    """
    # Tokenize the documents
    tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    new_queries = []

    for qi, q in enumerate(queries):
        new_queries.append((q, documents, bm25, top_k))

    # Use process_map from tqdm for parallel processing and progress display
    results = process_map(process_query, new_queries, max_workers=4, chunksize=1)
    return results

def bm25_similarity_multiprocess_dynamic(documents: List[str], queries: List[str], top_k: int, schema_map: Dict[str, int], last_retrieved_data) -> List[List[Tuple[int, float]]]:
    """
    Calculate BM25 similarity for multiple queries against all documents using multiple processes.

    :param documents: List of documents.
    :param queries: List of queries.
    :param top_k: Number of top documents to return for each query.
    :return: A list where each element corresponds to a query and contains a list of top_k documents and their scores.
    """
    # Tokenize the documents
    # tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
    # bm25 = BM25Okapi(tokenized_docs)

    # Define a helper function for parallel processing
    def process_query(query):
        qi = queries.index(query)
        docs_temp = [documents[schema_map[x]] for x in [ret["schema"] for ret in last_retrieved_data[qi]["retrieved"]]]
        return sorted(calculate_bm25_similarity(docs_temp, query, BM25Okapi([word_tokenize(doc.lower()) for doc in docs_temp])), key=lambda x: x[1], reverse=True)[:top_k]

    # Use process_map from tqdm for parallel processing and progress display
    results = process_map(process_query, queries, chunksize=1)
    return results


def extract_prediction(prediction: List[str], typee: str) -> str:
    if typee == "md":
        return "\n".join(prediction)
    else:
        return "\n".join(prediction).split("\n\n")[0]
    

def bm25_similarity_multiprocess_diff_docs(documents: List[List[str]], queries: List[str], top_k: int) -> List[List[Tuple[int, float]]]:
    """
    Calculate BM25 similarity for multiple queries, where each query corresponds to its own set of documents.

    :param documents: A list of document lists, where each inner list contains documents corresponding to a query.
    :param queries: A list of queries.
    :param top_k: The number of top documents to return for each query.
    :return: A list where each element corresponds to a query and contains a list of top_k documents and their scores.
    """
    new_queries = []

    for qi, q in enumerate(queries):
        # Tokenize the documents for the current query
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents[qi]]
        bm25 = BM25Okapi(tokenized_docs)
        
        # Append the query and its corresponding documents
        new_queries.append((q, documents[qi], bm25, top_k))

    # Use process_map from tqdm for parallel processing and progress display
    results = process_map(process_query, new_queries, max_workers=4, chunksize=1)
    
    return results



def compute_recall(pred_list: List[str], gold_list: List[str]) -> float:
    correct_count = 0
    for p in pred_list:
        if p in gold_list:
            correct_count += 1
    rec = 1.0*correct_count/len(gold_list)
    return rec


def compute_recall_multiple(top_k: List[int], pred_list: List[str], gold_list: List[str]) -> Dict[int, float]:
    # return {k: compute_recall(pred_list[:k], gold_list) for k in top_k if k < len(pred_list)}
    return {k: compute_recall(pred_list[:k], gold_list) for k in top_k if k <= len(pred_list) and len(gold_list) > 0}


def compute_em_multiple(top_k: List[int], pred_list: List[str], gold_list: List[str]) -> Dict[int, float]:
    em: Dict[int, float] = {k:0.0 for k in top_k}
    for k in top_k: 
        if set(gold_list).issubset(set(pred_list[:k])):
            em[k] += 1
    return em

# def evaluate_retrieve(top_k: List[int], data: List[Dict[str, Any]]) -> Dict[str,Dict[int, float]]:
#     recall: Dict[int, float] = {k:0.0 for k in top_k}
#     for di, d in enumerate(data):
#         rec = compute_recall_multiple(top_k, [x["schema"] for x in d["retrieved"]], d["gold"])
#         for k in top_k:
#             recall[k] += rec[k]
#     recall = {k: v/len(data) for k, v in recall.items()}
#     return {"recall": recall}

def evaluate_recall(top_k: List[int], pred_list: List[List[str]], gold_list: List[List[str]]) -> Dict[int, float]:
    recall: Dict[int, List[float, int]] = {k:[0.0, 0] for k in top_k}
    single_recall = []
    for pi, p in enumerate(pred_list):
        rec = compute_recall_multiple(top_k, p, gold_list[pi])
        single_recall.append(rec)
        for k in top_k:
            recall[k][0] += rec[k] if k in rec.keys() else 0
            recall[k][1] += 1 if k in rec.keys() else 0
    recall = {k: v[0]/v[1] for k, v in recall.items() if v[1] > 0}
    return recall, single_recall