# -*- coding: utf-8 -*-
"""

@author: mjayant
"""

from fastapi import FastAPI, UploadFile, File
from src.pipeline import run_pipeline

app = FastAPI(title="SemantiVenue AI API")

@app.post("/analyze")
async def analyze_paper(file: UploadFile = File(None), arxiv_id: str = None):
    if file:
        # Extend this for file handling if needed
        pass
    result = run_pipeline(arxiv_id or "", is_arxiv=bool(arxiv_id))
    return result