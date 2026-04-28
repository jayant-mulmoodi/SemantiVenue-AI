# -*- coding: utf-8 -*-
"""


@author: mjayant
"""

import sys
import langchain_core
import langchain_ollama
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

def verify_setup():
    print("--- System Check ---")
    print(f"Python Version: {sys.version}")
    print(f"LangChain Core: {langchain_core.__version__}")
    print(f"LangChain Ollama: {langchain_ollama.__version__}")
    
    print("\n--- Testing Chat Model (Llama 3.1) ---")
    try:
        # Initialize the model using your local name from ollama list
        llm = ChatOllama(
            model="llama3.1:latest", 
            temperature=0,
            validate_model_on_init=True 
        )
        
        response = llm.invoke("Hello, verify system status.")
        print("Llama 3.1 Response success.")
        print(f"Response: {response.content}")
        
    except Exception as e:
        print(f"Chat Model Error: {e}")

    print("\n--- Testing Embedding Model (Nomic) ---")
    try:
        # Testing your nomic embeddings for the RAG pipeline
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        vector = embeddings.embed_query("Testing semantic venue vectorization.")
        print(f"Embedding success. Vector length: {len(vector)}")
        
    except Exception as e:
        print(f"Embedding Error: {e}")

if __name__ == "__main__":
    verify_setup()
    
'''
arXiv ID,Paper Topic,Expected Top Conferences
2405.12345,Large Language Models,"NeurIPS, ICML, ICLR"
2305.18290,Vision Transformer / Computer Vision,"CVPR, ICCV, NeurIPS"
2410.02524,Reinforcement Learning,"NeurIPS, ICML"
2401.08234,NLP / Multimodal Models,"ACL, EMNLP"
2309.16609,General AI / Survey Paper,"AAAI, NeurIPS"

'''    