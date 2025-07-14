
import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    logger.info(f"Attempting to load documents from folder: {folder_path}")
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.endswith('.pdf'):
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    logger.info(f"Successfully loaded {len(docs)} documents from {filename}")
                except Exception as e:
                    logger.error(f"Error loading PDF file {filename}: {e}")
            else:
                logger.warning(f"Skipping unsupported file type: {filename}")
                continue
        logger.info(f"Finished loading documents. Total documents loaded: {len(documents)}")
        return documents
    except FileNotFoundError:
        logger.error(f"Folder not found: {folder_path}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading documents: {e}")
        return []

folder_path = "./data"
try:
    documents = load_documents(folder_path)
    logger.info(f"Loaded {len(documents)} documents from the folder.")
except Exception as e:
    logger.critical(f"Failed to load documents at application startup: {e}")
    documents = [] # Ensure documents is defined even if loading fails
