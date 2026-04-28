import json
import os
import logging
import warnings
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def get_chroma_path():
    """Adaptive path: /tmp on Streamlit Cloud, local folder on Windows"""
    if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
        return "/tmp/chroma_db"      # Cloud
    return "chroma_db"               # Local

def build_vector_db():
    logger.info("🚀 Building Vector Database...")

    chroma_dir = Path(get_chroma_path())
    
    # Clean old database to avoid dimension mismatch
    if chroma_dir.exists():
        import shutil
        shutil.rmtree(chroma_dir)
        logger.info("🗑️ Old database removed.")

    chroma_dir.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-base-en-v1.5"
    )

    collection = client.get_or_create_collection(
        name="conferences",
        embedding_function=embedding_function
    )

    with open("data/conferences.json", "r", encoding="utf-8") as f:
        conferences = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for i, conf in enumerate(conferences):
        doc = f"Conference: {conf['name']}\nScope: {conf['description']}\nTopics: {', '.join(conf.get('topics', []))}"
        documents.append(doc)
        metadatas.append({"name": conf["name"], "website": conf.get("website", "")})
        ids.append(f"conf_{i}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    logger.info(f"✅ Vector database built successfully with {len(conferences)} conferences!")

if __name__ == "__main__":
    build_vector_db()