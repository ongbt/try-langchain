 

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.chroma import Chroma
import os
import shutil

  

def load_documents(data_path):
    loader = DirectoryLoader(data_path, {
        ".pdf": lambda path: PyPDFLoader(path), 
        ".csv": lambda path: CSVLoader(path, "text"), 
    })
    loader = PyPDFDirectoryLoader(data_path)

    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[0]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document], chroma_path):
    # Clear out the database first.
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OllamaEmbeddings(), persist_directory=chroma_path
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {chroma_path}.")


 