import fitz
import arxiv
import logging
from typing import Dict

logger = logging.getLogger(__name__)

def parse_pdf(pdf_path: str) -> str:
    logger.debug(f"Parsing PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()

def fetch_arxiv_paper(arxiv_id: str) -> Dict:
    logger.info(f"Fetching arXiv paper: {arxiv_id}")
    search = arxiv.Search(id_list=[arxiv_id], max_results=1)
    paper = next(search.results())
    return {
        "title": paper.title,
        "abstract": paper.summary,
        "text": f"Title: {paper.title}\nAbstract: {paper.summary}"
    }

def process_input(input_path: str, is_arxiv: bool = False) -> Dict:
    if is_arxiv:
        return fetch_arxiv_paper(input_path)
    else:
        text = parse_pdf(input_path)
        lines = text.splitlines()
        title = lines[0].strip() if lines else "Untitled Paper"
        abstract = " ".join(lines[1:300])[:2000]
        logger.info(f"Processed paper: {title}")
        return {"title": title, "abstract": abstract, "text": text[:8000]}