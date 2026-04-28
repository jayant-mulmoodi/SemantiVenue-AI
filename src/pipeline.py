import logging
from src.graph import agentic_graph
from src.evaluation_metrics import evaluate_ranking_performance

logger = logging.getLogger(__name__)

def run_pipeline(input_path: str, is_arxiv: bool = False):
    logger.info(f"Starting full agentic pipeline for: {input_path}")
    
    initial_state = {
        "input_path": input_path,
        "is_arxiv": is_arxiv
    }
    
    result = agentic_graph.invoke(initial_state)
    
    # Compute NDCG and MRR
    ranking_metrics = evaluate_ranking_performance(
        result["ranked_docs"],
        result["ranked_scores"]
    )
    
    logger.info("Pipeline completed successfully")
    return {
        "paper_title": result["paper"]["title"],
        "ranked_conferences": result["ranked_docs"],
        "scores": result["ranked_scores"],
        "explanation": result["evaluation"],
        "metrics": ranking_metrics
    }