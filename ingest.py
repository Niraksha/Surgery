import os
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings
from langchain.document_loaders import PyPDFLoader  # Use PyPDFLoader for PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add your Cohere API key directly here
cohere_api_key = "nOumjlER154XIqtpAE5JADBEyeSz2Kblhn3SLxT4"

def ingest_docs():
    # Define the path to your docs folder
    docs_folder = "docs"
    
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(docs_folder) if f.endswith('.pdf')]

    all_documents = []

    # Loop through each PDF file and load it
    for pdf_file in pdf_files:
        pdf_path = os.path.join(docs_folder, pdf_file)
        print(f"📄 Loading PDF: {pdf_path}")
        
        # Load the PDF document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        all_documents.extend(documents)  # Add the documents to the list

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = text_splitter.split_documents(all_documents)

    print(f"📄 Loaded {len(all_documents)} pages from {len(pdf_files)} PDFs")
    print(f"✂️ Split into {len(chunks)} chunks")

    # Set up the Cohere embeddings
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)

    # Create the FAISS vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("✅ Vectorstore created successfully.")

    # Save the vectorstore to disk
    vectorstore.save_local("cabg_faiss_index")
    print("💾 Vectorstore saved to 'cabg_faiss_index'")

if __name__ == "__main__":
    ingest_docs()
