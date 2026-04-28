import chromadb
import os
import logging
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

def get_chroma_path():
    """Same adaptive path as build_vector_db.py"""
    if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
        return "/tmp/chroma_db"
    return "chroma_db"

def retrieve_candidates(query_text: str, top_k: int = 15):
    logger.info(f"Retrieving top {top_k} candidates...")

    client = chromadb.PersistentClient(path=get_chroma_path())
    
    # Force the SAME embedding model as build
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5"
    )

    collection = client.get_or_create_collection(
        name="conferences",
        embedding_function=embedding_function
    )

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

    logger.info(f"Retrieved {len(results['documents'][0])} candidates successfully")
    return {
        "documents": results["documents"][0],
        "distances": results["distances"][0]
    }