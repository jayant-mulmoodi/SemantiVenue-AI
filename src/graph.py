import logging
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

from src.paper_processor import process_input
from src.retriever import retrieve_candidates
from src.reranker import rerank_candidates
from src.evaluator import evaluate_with_llm

logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    input_path: str
    is_arxiv: bool
    paper: dict
    query: str
    retrieved_docs: List[str]
    retrieved_scores: List[float]
    ranked_docs: List[str]
    ranked_scores: List[float]
    evaluation: str

def parse_node(state: GraphState):
    logger.info("Step 1: Parsing paper...")
    paper = process_input(state["input_path"], state["is_arxiv"])
    query = f"{paper['title']} {paper.get('abstract', '')}"
    return {"paper": paper, "query": query}

def retrieve_node(state: GraphState):
    logger.info("Step 2: Semantic retrieval...")
    results = retrieve_candidates(state["query"])
    return {
        "retrieved_docs": results["documents"],
        "retrieved_scores": results["distances"]
    }

def rerank_node(state: GraphState):
    logger.info("Step 3: Cross-encoder re-ranking...")
    ranked_docs, ranked_scores = rerank_candidates(
        state["query"], 
        state["retrieved_docs"], 
        state["retrieved_scores"]
    )
    return {
        "ranked_docs": ranked_docs,
        "ranked_scores": ranked_scores
    }

def evaluate_node(state: GraphState):
    logger.info("Step 4: LLM Evaluation...")
    evaluation = evaluate_with_llm(
        state["paper"]["title"],
        state["paper"].get("abstract", ""),
        state["ranked_docs"],
        state["ranked_scores"]
    )
    return {"evaluation": evaluation}

# Build LangGraph
workflow = StateGraph(GraphState)
workflow.add_node("parse", parse_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("evaluate", evaluate_node)

workflow.add_edge(START, "parse")
workflow.add_edge("parse", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "evaluate")
workflow.add_edge("evaluate", END)

agentic_graph = workflow.compile()
logger.info("LangGraph agentic workflow compiled successfully")