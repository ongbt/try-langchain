from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

import sys
sys.setrecursionlimit(10**6) 
DB_FAISS_PATH = "vectorstore/db_faiss"
loader = CSVLoader(file_path="titanic.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
print(data)

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

print(len(text_chunks))

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)

docsearch.save_local(DB_FAISS_PATH)


#query = "What is the value of GDP per capita of Finland provided in the data?"

#docs = docsearch.similarity_search(query, k=3)

#print("Result", docs)

# llm = CTransformers(model="models/llama-2-7b-chat.ggmlv3.q4_0.bin",
#                     model_type="llama",
#                     max_new_tokens=512,
#                     temperature=0.1)
llm = Ollama(model="llama2")
qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

while True:
    chat_history = []
    #query = "What is the value of  GDP per capita of Finland provided in the data?"
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])