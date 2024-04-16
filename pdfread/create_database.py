from lib import common
import shutil
import os
import sys
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

import argparse

# from langchain.embeddings import OpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

# getting the name of the directory where the this file is present.
# Getting the parent directory name where the current directory is present.
# adding the parent directory to the sys.path.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


CHROMA_PATH = "chroma/pdfread/"
DATA_PATH = "data/"


def main():
    generate_data_store()


def generate_data_store():

    topic = common.get_topic()

    data_path = DATA_PATH + topic
    print(f"Loading Data from {data_path}")
    documents = common.load_documents(data_path)
    chunks = common.split_text(documents)
    print("Saving to Database...")

    chroma_path = CHROMA_PATH + topic
    common.save_to_chroma(chunks, chroma_path)


if __name__ == "__main__":
    main()


# python -m pdfread.create_database "xian_tour"
