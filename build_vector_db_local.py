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
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

def build_vector_db():
    logger.info("Starting to build vector database...")

    chroma_dir = Path("chroma_db")
    if chroma_dir.exists():
        import shutil
        shutil.rmtree(chroma_dir)
        logger.info("Old chroma_db folder deleted successfully.")

    chroma_dir.mkdir(exist_ok=True)

    settings = chromadb.Settings(anonymized_telemetry=False, allow_reset=True)
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=settings)

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
    logger.info(f" Vector database built successfully with {len(conferences)} conferences")

if __name__ == "__main__":
    build_vector_db()