import os
import numpy as np
import logging
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2"),
    temperature=float(os.getenv("OLLAMA_TEMPERATURE", 0.3)),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

prompt_template = ChatPromptTemplate.from_template("""
You are an expert academic conference advisor.

Paper Title: {title}
Paper Abstract: {abstract}

Candidate Conferences (with scores):
{conference_list}

Rank the top 3 most suitable conferences. For each:
- Explain semantic alignment with the paper
- Highlight strengths and potential acceptance factors
- Suggest specific improvements to increase acceptance chance
- Note any risks or mismatches

Keep the response professional, concise, and actionable.
""")

def evaluate_with_llm(title: str, abstract: str, conferences: list, scores: list) -> str:
    logger.info("Running LLM evaluation...")
    
    conf_list = "\n".join(
        [f"{name} (score: {score:.3f})\nDescription: {desc}" 
         for name, desc, score in zip(conferences, conferences, scores)]
    )
    
    chain = prompt_template | llm
    response = chain.invoke({
        "title": title,
        "abstract": abstract,
        "conference_list": conf_list
    })
    
    logger.info("LLM evaluation completed")
    return response.content
