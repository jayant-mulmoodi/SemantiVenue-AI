import sys
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv
import subprocess


import streamlit as st


# ====================== MUST BE FIRST ======================
# Set page config as the VERY FIRST Streamlit command
st.set_page_config(page_title="SemantiVenue AI", layout="wide")
# ==========================================================

# ====================== Fixes ======================
warnings.filterwarnings("ignore")
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
torch.classes.__path__ = []
# ===================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# ====================== CRITICAL PATH FIX FOR STREAMLIT CLOUD ======================
# Force add project root to Python path
#ROOT_DIR = Path(__file__).parent.parent.absolute()
#sys.path.insert(0, str(ROOT_DIR))
#sys.path.insert(0, str(ROOT_DIR / "src"))

#print(f"Project root added to path: {ROOT_DIR}")
# =================================================================================

load_dotenv()

import logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

import streamlit as st
from src.pipeline import run_pipeline

# ====================== CHROMADB INITIALIZATION ======================
@st.cache_resource(show_spinner="Building Vector Database (First Time)...")
def initialize_chroma_db():
    """Build ChromaDB on startup"""
    try:
        logger.info("Initializing ChromaDB...")
        result = subprocess.run(
            ["python", "build_vector_db.py"], 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        if result.returncode == 0:
            logger.info("ChromaDB initialized successfully")
            return True
        else:
            logger.error(f"DB build failed: {result.stderr}")
            st.error("Failed to build vector database. Please check logs.")
            return False
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        st.error(f"Database error: {e}")
        return False

# Run initialization
db_ready = initialize_chroma_db()
# ===================================================================

st.title("SemantiVenue AI")
st.markdown("**Agentic RAG Research Paper Conference Recommendation System**")

tab1, tab2 = st.tabs(["Submit Paper", "Results"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload research paper (PDF)", type=["pdf"])
    with col2:
        arxiv_id = st.text_input("Or enter arXiv ID (e.g., 2405.12345)")

    if st.button("Analyze", type="primary"):
        if uploaded_file or arxiv_id:
            with st.spinner("Running Agentic RAG Pipeline..."):
                try:
                    if uploaded_file:
                        temp_dir = Path("/tmp")
                        temp_dir.mkdir(exist_ok=True)
                        temp_path = temp_dir / "temp_upload.pdf"
                        temp_path.write_bytes(uploaded_file.getvalue())
                        result = run_pipeline(str(temp_path))
                        temp_path.unlink(missing_ok=True)
                    else:
                        result = run_pipeline(arxiv_id, is_arxiv=True)
                    
                    st.session_state.result = result
                    st.success("✅ Analysis completed successfully")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    if "result" in st.session_state:
        r = st.session_state.result
        st.subheader(f"Paper: {r['paper_title']}")

        st.write("### 📊 Retrieval & Ranking Performance Metrics")
        metrics = r.get("metrics", {})
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("NDCG@5", f"{metrics.get('ndcg@5', 0):.4f}")
            with col2:
                st.metric("MRR", f"{metrics.get('mrr', 0):.4f}")
            with col3:
                st.metric("Top-1 Score", f"{metrics.get('top_1_score', 0):.4f}")
            with col4:
                st.metric("Avg Fusion Score", f"{metrics.get('avg_fusion_score', 0):.4f}")

        st.write("### 🏆 Ranked Conferences")
        for i, (conf, score) in enumerate(zip(r["ranked_conferences"], r["scores"])):
            with st.expander(f"Rank {i+1}: {conf} (Score: {score:.3f})", expanded=(i == 0)):
                st.text_area(
                    label="Detailed Recommendation",
                    value=r["explanation"],
                    height=340,
                    disabled=True,
                    key=f"rec_{i}"
                )

        st.download_button(
            label="Download Full Report",
            data=r["explanation"],
            file_name="conference_recommendation_report.txt",
            mime="text/plain"
        )