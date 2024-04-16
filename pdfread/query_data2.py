
from lib import common

import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import sys

# getting the name of the directory where the this file is present.
# Getting the parent directory name where the current directory is present.
# adding the parent directory to the sys.path.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


CHROMA_PATH = "chroma/pdfread/"

PROMPT_TEMPLATE = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """


def interactive(root_chroma_path, prompt_template):

    model = Ollama()

    topic = common.get_topic()
    chroma_path = root_chroma_path + topic

    # Prepare the DB.
    embedding_function = OllamaEmbeddings(model="llama2:latest")
    db = Chroma(persist_directory=chroma_path,
                embedding_function=embedding_function)
    prompt_template = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"])

    chain_type_kwargs = {"prompt": prompt_template}
    qa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs)

    while (True):
        query_text = input("Enter query: ")

        # print("Query is: " + query_text)

        results = qa({"query": query_text})
        # print("results")
        if len(results) == 0:
            # or results[0][1] < 0.1:
            print(f"Unable to find matching results.")
            return

        print("Response : ", results["result"])

        print("\n")


if __name__ == "__main__":
    # main()
    interactive(CHROMA_PATH, PROMPT_TEMPLATE)


# python -m pdfread.query_data "xian_tour"     
    # Difference between Data Security and Cybersecurity
