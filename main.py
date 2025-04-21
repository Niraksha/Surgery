import faiss
import os
import numpy as np
import fitz  # PyMuPDF for PDF text extraction
from sentence_transformers import SentenceTransformer

# Path to the FAISS index and directory with PDFs
index_path = r"C:\Users\Express\OneDrive\Desktop\RAG MODEL\cabg_faiss_index"
pdfs_directory = r"C:\Users\Express\OneDrive\Desktop\RAG MODEL\docs"  # Directory containing PDF files

# Check if the FAISS index exists
if not os.path.exists(index_path):
    print(f"Error: The FAISS index file at {index_path} does not exist.")
    exit()

# Load the FAISS index
try:
    print("Attempting to load the FAISS index...")
    index = faiss.read_index(index_path)
    print("FAISS index loaded successfully!")
except Exception as e:
    print(f"An error occurred while loading the FAISS index: {e}")
    exit()

# Function to extract text from a PDF file using PyMuPDF (fitz)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()  # Extract text from each page
    return text

# Function to load PDFs from a directory and extract text
def load_documents_from_pdfs(directory):
    documents = []
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        text = extract_text_from_pdf(pdf_path)
        documents.append(text)
    return documents

# Load all documents (PDFs) from the specified directory
print("Loading documents from PDFs...")
documents = load_documents_from_pdfs(pdfs_directory)
print(f"Loaded {len(documents)} documents from PDFs.")

# Initialize the transformer model for encoding queries and documents
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to query the FAISS index
def query_faiss(query_vector, k=5):
    # Search the FAISS index
    D, I = index.search(np.array([query_vector], dtype=np.float32), k)
    return D, I

# Prompt the user for a query (for simplicity, just a basic text input)
query = input("Enter your query: ")

# Convert the query to a vector using the transformer model
query_vector = model.encode(query).tolist()

# Perform the query
print(f"Searching for the query: '{query}'...")
distances, indices = query_faiss(query_vector, k=5)

# Display the results
print("\nTop 5 most similar documents:")
for idx, dist in zip(indices[0], distances[0]):
    print(f"\nDocument ID: {idx}")
    print(f"Distance: {dist}")
    print(f"Document Content: {documents[idx][:500]}...")  # Show a snippet of the document (first 500 characters)
