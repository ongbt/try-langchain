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
import sys
# from dotenv import load_dotenv

# load_dotenv() 

# getting the name of the directory where the this file is present.
# Getting the parent directory name where the current directory is present.
# adding the parent directory to the sys.path.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import common 

CHROMA_PATH = "../chroma/pdf-read"
DATA_PATH = "../data/books"


def main():
    generate_data_store()


def generate_data_store():
    documents = common.load_documents(DATA_PATH)
    chunks = common.split_text(documents)
    common.save_to_chroma(chunks, CHROMA_PATH)



if __name__ == "__main__":
    main()