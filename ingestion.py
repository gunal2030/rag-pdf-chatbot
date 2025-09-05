import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_data(file_path: str):
    """
    Loads, splits, embeds, and stores documents in a FAISS vector store.
    """
    print("Starting data ingestion...")
    # Load the document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split document into {len(docs)} chunks.")

    # Create embeddings using a Sentence-Transformer model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Store embeddings in a FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store_path = "faiss_index"
    vector_store.save_local(vector_store_path)
    print(f"Vector store created and saved to '{vector_store_path}'.")
    return vector_store_path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(current_dir, "sample.pdf")
    ingest_data(pdf_path)