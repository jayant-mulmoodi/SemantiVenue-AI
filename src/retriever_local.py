import chromadb
import os
import logging
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

load_dotenv()
logger = logging.getLogger(__name__)

def retrieve_candidates(query_text: str, top_k: int = 20):
    logger.info(f"Retrieving top {top_k} candidates...")

    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "chroma_db"))
    
    # Force same embedding model as build
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
    
    logger.info(f"Successfully retrieved {len(results['documents'][0])} candidates")
    return {
        "documents": results["documents"][0],
        "distances": results["distances"][0]
    }