
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

folder_path = "./data"
documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents from the folder.")
