import chromadb
import os
import logging
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

load_dotenv()
logger = logging.getLogger(__name__)

def get_chroma_path():
    if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
        return "/tmp/chroma_db"
    return "chroma_db"

def retrieve_candidates(query_text: str, top_k: int = 15):
    logger.info(f"Retrieving top {top_k} candidates...")
    
    client = chromadb.PersistentClient(path=get_chroma_path())
    collection = client.get_collection(os.getenv("CHROMA_COLLECTION_NAME", "conferences"))

    results = collection.query(query_texts=[query_text], n_results=top_k)
    return {
        "documents": results["documents"][0],
        "distances": results["distances"][0]
    }