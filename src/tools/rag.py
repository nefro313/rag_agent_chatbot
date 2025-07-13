import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from src.utils.document_loader import documents

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
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {SAVE_PATH}")
        # Load the existing vector store from the specified folder
        vectordb = FAISS.load_local(
            SAVE_PATH, EMBEDDINGS, allow_dangerous_deserialization=True
        )
    else:
        print(
            f"No FAISS index found at {FAISS_INDEX_PATH}. Creating a new one.")
        # Step 1: Create the vector store from documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = text_splitter.split_documents(documents)
        print(f"Split the source documents into {len(chunks)} chunks.")

        vectordb = FAISS.from_documents(chunks, EMBEDDINGS)

        # Step 2: Save the newly created vector store to disk
        vectordb.save_local(SAVE_PATH)
        print(f"Successfully created and saved FAISS index to {SAVE_PATH}")

    return vectordb


# --- INITIALIZATION ---
# This code runs ONCE when the module is first imported.
# It creates/loads the vector store and the retriever, making them ready for use.
print("Initializing RAG tool...")
vectordb = initialize_vector_store()
retriever = vectordb.as_retriever(search_kwargs={"k": 2})
print("RAG tool initialized successfully.")


# --- TOOL DEFINITION ---
@tool
def rag_search_tool(query: str) -> str:
    """
    Searches the RAG knowledge base for relevant documents and returns their content.
    """
    try:
        # The retriever is already initialized and ready to be used.
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs) if docs else "No relevant documents found."
    except Exception as e:
        return f"RAG_ERROR::{e}"
