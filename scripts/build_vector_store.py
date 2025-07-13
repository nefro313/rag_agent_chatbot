import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from  langchain_chroma import Chroma 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.utils.document_loader import documents

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
print(f"Split the documents into {len(chunks)} chunks.")
# ── Config ────────────────────────────────────────────────────────────
SOURCE_DIR   = Path("data")            
INDEX_DIR    = Path("vector_store/chroma_db")        
COLLECTION   = "kb_collection"
EMBED_MODEL  = "text-embedding-3-small"

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectordb   = Chroma.from_documents(
    documents         = chunks,
    embedding         = embeddings,
    persist_directory = str(INDEX_DIR),
    collection_name   = COLLECTION,
)
print("Index built at", INDEX_DIR.resolve())
