from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

INDEX_DIR = "vector_store/chroma_db"
COLLECTION = "kb_collection"
EMBED_MODEL = "text-embedding-3-small"

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectordb = Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings, collection_name=COLLECTION)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

@tool
def rag_search_tool(query: str) -> str:
    """Top-3 chunks from KB (empty string if none)"""
    try:
        docs = retriever.invoke(query, k=3)
        return "\n\n".join(d.page_content for d in docs) if docs else ""
    except Exception as e:
        return f"RAG_ERROR::{e}"
