import sys
import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# ====================== Streamlit & Torch Fixes ======================
warnings.filterwarnings("ignore")
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import torch
torch.classes.__path__ = []   # Fix for torch RuntimeError
# =====================================================================

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

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

st.set_page_config(page_title="SemantiVenue AI", layout="wide")
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
            with st.spinner("Running Agentic RAG Pipeline (Parse → Retrieve → Re-rank → LLM Evaluate)..."):
                try:
                    if uploaded_file:
                        # FIXED: Use proper temporary path for Windows
                        temp_dir = Path("temp")
                        temp_dir.mkdir(exist_ok=True)
                        temp_path = temp_dir / "temp_upload.pdf"
                        
                        temp_path.write_bytes(uploaded_file.getvalue())
                        result = run_pipeline(str(temp_path))
                        temp_path.unlink(missing_ok=True)
                    else:
                        result = run_pipeline(arxiv_id, is_arxiv=True)
                    
                    st.session_state.result = result
                    st.success("--> Analysis completed successfully")
                    logger.info("Analysis completed successfully")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    logger.error(f"Pipeline failed: {str(e)}")

with tab2:
    if "result" in st.session_state:
        r = st.session_state.result
        st.subheader(f"Paper: {r['paper_title']}")

        # ====================== METRICS SECTION ======================
        st.write("###  Retrieval & Ranking Performance Metrics")
        metrics = r.get("metrics", {})
        
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("NDCG@5", f"{metrics.get('ndcg@5', 0):.4f}", help="Higher is better (1.0 = perfect ranking)")
            with col2:
                st.metric("MRR", f"{metrics.get('mrr', 0):.4f}", help="Mean Reciprocal Rank")
            with col3:
                st.metric("Top-1 Raw Score", f"{metrics.get('top_1_score', 0):.4f}", help="Raw Cross-Encoder score")
            with col4:
                st.metric("Normalized Top-1", f"{metrics.get('normalized_top_1', 0):.4f}", help="After normalization [0-1]")

            st.caption("**Note**: Raw scores can be negative. NDCG and MRR use normalized scores.")
            st.caption(metrics.get("agentic_rag_note", ""))
        else:
            st.info("Metrics will appear here after analysis.")
        # ============================================================

        st.write("###  Ranked Conferences")
        for i, (conf, score) in enumerate(zip(r["ranked_conferences"], r["scores"])):
            with st.expander(f"Rank {i+1}: {conf} (Score: {score:.3f})", expanded=(i == 0)):
                st.text_area(
                    label="Detailed Recommendation",
                    value=r["explanation"],
                    height=340,
                    disabled=True,
                    key=f"recommendation_{i}"
                )

        st.download_button(
            label="Download Full Report",
            data=r["explanation"],
            file_name="conference_recommendation_report.txt",
            mime="text/plain"
        )