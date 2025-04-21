import os
from dotenv import load_dotenv
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Cohere

# Load environment variables from .env file
load_dotenv(dotenv_path="C:\Users\Express\OneDrive\Desktop\RAG MODEL\open.env")

def load_rag_chain():
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("Cohere API key not found. Please set it in the open.env file.")

    user_agent = "surgery"

    # Use updated CohereEmbeddings without directly passing api_key
    embeddings = CohereEmbeddings(model="embed-english-v3.0", user_agent=user_agent)

    # Fix the deserialization error by enabling dangerous deserialization (safe if it's your own index)
    vectorstore = FAISS.load_local(
        "cabg_faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever()

    llm = Cohere(model="command-nightly", temperature=0.3)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return rag_chain
