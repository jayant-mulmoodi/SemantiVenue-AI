import json
import os
import logging
import warnings
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def get_chroma_path():
    """Return correct ChromaDB path for Streamlit Cloud or Local Windows"""
    # Streamlit Cloud uses /tmp (writable)
    if os.path.exists("/tmp") and os.access("/tmp", os.W_OK):
        path = "/tmp/chroma_db"
        logger.info("Using Cloud path: /tmp/chroma_db")
    else:
        path = "chroma_db"
        logger.info("Using Local path: chroma_db")
    return path

def build_vector_db():
    logger.info("🚀 Starting Vector Database Build...")

    chroma_dir = Path(get_chroma_path())
    
    # Clean previous database
    if chroma_dir.exists():
        import shutil
        shutil.rmtree(chroma_dir)
        logger.info("🗑️ Old database cleaned.")
    
    chroma_dir.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )

    collection = client.get_or_create_collection(
        name=os.getenv("CHROMA_COLLECTION_NAME", "conferences"),
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