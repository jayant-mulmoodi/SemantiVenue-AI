# SemantiVenue AI

Agentic Research Paper to Conference Recommendation System using LangGraph.

## Setup

1. Install Ollama and pull model: `ollama pull llama3.2`
2. `pip install -r requirements.txt`
3. `python build_vector_db.py`
4. `streamlit run app/streamlit_app.py`

## Features
- PDF and arXiv input support
- Embedding retrieval + Cross-encoder re-ranking
- LangGraph agentic workflow
- Local LLM evaluation with explanations