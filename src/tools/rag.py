import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from src.utils.document_loader import documents

# ── LOGGING ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
SAVE_PATH = "vector_store"
# FAISS automatically creates index.faiss and index.pkl
FAISS_INDEX_PATH = os.path.join(SAVE_PATH, "index.faiss")
EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

def initialize_vector_store():
    """
    Checks if a FAISS index exists on disk. If not, it creates one from the source documents.
    It then returns the loaded or newly created FAISS vector store.
    """
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            logger.info(f"Loading existing FAISS index from {SAVE_PATH}")
            # Load the existing vector store from the specified folder
            vectordb = FAISS.load_local(
                SAVE_PATH, EMBEDDINGS, allow_dangerous_deserialization=True
            )
            logger.info("FAISS index loaded successfully.")
        else:
            logger.info(f"No FAISS index found at {FAISS_INDEX_PATH}. Creating a new one.")
            # Step 1: Create the vector store from documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split the source documents into {len(chunks)} chunks.")

            vectordb = FAISS.from_documents(chunks, EMBEDDINGS)

            # Step 2: Save the newly created vector store to disk
            vectordb.save_local(SAVE_PATH)
            logger.info(f"Successfully created and saved FAISS index to {SAVE_PATH}")

        return vectordb
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

# --- INITIALIZATION ---
# This code runs ONCE when the module is first imported.
# It creates/loads the vector store and the retriever, making them ready for use.
try:
    logger.info("Initializing RAG tool...")
    vectordb = initialize_vector_store()
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    logger.info("RAG tool initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize RAG tool: {e}")
    # Depending on desired behavior, you might want to re-raise or handle differently
    raise


# --- TOOL DEFINITION ---
@tool
def rag_search_tool(query: str) -> str:
    """
    Searches the RAG knowledge base for relevant documents and returns their content.
    """
    try:
        logger.info(f"Performing RAG search for query: {query}")
        # The retriever is already initialized and ready to be used.
        docs = retriever.invoke(query)
        if docs:
            logger.info(f"Found {len(docs)} relevant documents for query: {query}")
            return "\n\n".join(d.page_content for d in docs)
        else:
            logger.info(f"No relevant documents found for query: {query}")
            return "No relevant documents found."
    except Exception as e:
        logger.error(f"Error during RAG search for query '{query}': {e}")
        return f"RAG_ERROR::{e}"