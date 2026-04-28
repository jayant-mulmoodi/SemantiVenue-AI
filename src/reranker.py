import os
import logging
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()
logger = logging.getLogger(__name__)

# Fix for torch + streamlit compatibility
import torch
torch.classes.__path__ = []

cross_encoder = CrossEncoder(
    os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    device="cpu"
)

RETRIEVAL_WEIGHT = float(os.getenv("RETRIEVAL_WEIGHT", 0.4))
RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", 0.6))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", 5))

def rerank_candidates(query: str, candidates: list, retrieval_scores: list):
    logger.info("Performing cross-encoder re-ranking...")
    
    pairs = [[query, doc] for doc in candidates]
    rerank_scores = cross_encoder.predict(pairs)
    
    fused_scores = [RETRIEVAL_WEIGHT * r + RERANK_WEIGHT * rr 
                   for r, rr in zip(retrieval_scores, rerank_scores)]
    
    sorted_indices = np.argsort(fused_scores)[::-1]
    top_docs = [candidates[i] for i in sorted_indices[:FINAL_TOP_N]]
    top_scores = [fused_scores[i] for i in sorted_indices[:FINAL_TOP_N]]
    
    logger.info(f"Re-ranking completed. Selected top {FINAL_TOP_N} conferences.")
    return top_docs, top_scores