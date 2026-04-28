import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
logger = logging.getLogger(__name__)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

prompt_template = ChatPromptTemplate.from_template("""
You are an expert academic conference advisor.

Paper Title: {title}
Paper Abstract: {abstract}

Candidate Conferences:
{conference_list}

Rank the top 3 most suitable conferences. For each:
- Explain semantic alignment
- Highlight strengths and acceptance factors
- Suggest specific improvements
- Note any risks

Keep response professional, concise and actionable.
""")

def evaluate_with_llm(title: str, abstract: str, conferences: list, scores: list) -> str:
    logger.info("LLM evaluation started using Groq")
    conf_list = "\n".join([f"{name} (score: {score:.3f})" for name, score in zip(conferences, scores)])
    chain = prompt_template | llm
    response = chain.invoke({
        "title": title,
        "abstract": abstract,
        "conference_list": conf_list
    })
    return response.content