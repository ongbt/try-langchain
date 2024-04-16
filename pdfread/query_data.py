
from lib import common

import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate

import os
import sys

# getting the name of the directory where the this file is present.
# Getting the parent directory name where the current directory is present.
# adding the parent directory to the sys.path.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


CHROMA_PATH = "chroma/pdfread/xian_tour"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def get_query_text():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    return args.query_text


def interactive(chroma_path, prompt_template_text):

    # Prepare the DB.
    embedding_function = OllamaEmbeddings()
    db = Chroma(persist_directory=chroma_path,
                embedding_function=embedding_function)
    prompt_template = ChatPromptTemplate.from_template(
        prompt_template_text)

    model = Ollama()

    while (True):
        query_text = input("Enter query: ")

        # print("Query is: " + query_text)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=20)

        # print("results")
        if len(results) == 0:
            # or results[0][1] < 0.1:
            print(f"Unable to find matching results.")
            return

        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _score in results])
        prompt = prompt_template.format(
            context=context_text, question=query_text)
        # print(prompt)

        response_text = model.invoke(prompt)
        parse_response(response_text=response_text, results=results)

        print("\n\n")


def main():

    query_text = get_query_text()
    response_text, results = run(CHROMA_PATH, PROMPT_TEMPLATE, query_text)
    parse_response(response_text=response_text, results=results)


def parse_response(response_text, results):
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)


def run(chroma_path, prompt_template, query_text):
    # Prepare the DB.
    embedding_function = OllamaEmbeddings()
    db = Chroma(persist_directory=chroma_path,
                embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=20)

    # print("results")
    # print((results))
    if len(results) == 0:
        # or results[0][1] < 0.1:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = Ollama()

    response_text = model.invoke(prompt)

    return response_text, results


if __name__ == "__main__":
    # main()
    interactive(CHROMA_PATH, PROMPT_TEMPLATE)


# python query_data.py "data privacy protection?"
    # Difference between Data Security and Cybersecurity
