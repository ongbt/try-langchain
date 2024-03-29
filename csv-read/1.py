from langchain_community.llms import HuggingFaceTextGenInference 
from langchain.document_loaders.csv_loader import CSVLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain.chains import ConversationalRetrievalChain 

DB_FAISS_PATH = "vectorstore/db_faiss"
loader = CSVLoader(file_path="titanic.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()

# Split the text into Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(data)

# Download Sentence Transformers Embedding From Hugging Face
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

# COnverting the text Chunks into embeddings and saving the embeddings into FAISS Knowledge Base
docsearch = FAISS.from_documents(text_chunks, embeddings)

docsearch.save_local(DB_FAISS_PATH)

llm = HuggingFaceTextGenInference(
    inference_server_url="http://127.0.0.1:8080/",
    max_new_tokens=512,
    top_k=50,
    temperature=0.1,
    repetition_penalty=1.03,
)

qa = ConversationalRetrievalChain.from_llm(llm, retriever=docsearch.as_retriever())

exit = 0
while exit != 1:
    chat_history = []
    query = input(f"Input Prompt: ")
    if query == 'exit':
        print('Exiting')
        exit = 1
        break
    if query == '':
        continue
    result = qa({"question":query, "chat_history":chat_history})
    print("Response: ", result['answer'])