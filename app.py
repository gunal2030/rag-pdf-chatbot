import streamlit as st
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile

# Function to get the ConversationalRetrievalChain
def get_conversation_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        temperature=0, 
        api_key=st.secrets["GOOGLE_API_KEY"]
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return conversation_chain

# This is where we implement the Hybrid Search with Re-ranking
@st.cache_resource
def get_hybrid_retriever(documents):
    # Set the environment variable for Cohere to fix the API key error
    os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
    
    # Initialize retrievers
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_retriever = FAISS.from_documents(documents, embeddings).as_retriever(search_kwargs={"k": 10})

    # Initialize BM25 and Ensemble Retrievers
    bm25_retriever = BM25Retriever.from_documents(documents, search_kwargs={"k": 10})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], 
        weights=[0.5, 0.5]
    )

    # Initialize Cohere Re-ranker (no api_key argument needed here)
    compressor = CohereRerank(
        model="rerank-english-v3.0", 
        top_n=3
    )

    # Wrap the ensemble retriever with the re-ranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever
    )
    
    return compression_retriever

def handle_user_input(user_question):
    # Get the conversation chain from session_state
    if "conversation" not in st.session_state:
        st.error("Please upload a PDF first to start the chat.")
        return

    # Process the user question with the conversation chain
    response = st.session_state.conversation({'question': user_question, 'chat_history': st.session_state.chat_history})
    st.session_state.chat_history.append((user_question, response['answer']))
    st.session_state.source_documents = response.get('source_documents', [])

    # Display the conversation
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(bot_msg)

    # Display sources with a toggle
    if st.session_state.source_documents:
        with st.expander("See Sources"):
            for i, doc in enumerate(st.session_state.source_documents):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content)

def main():
    st.set_page_config(page_title="Chat with your PDF", page_icon=":books:")
    st.header("Chat with your PDF :books:")
    st.subheader("I am your personal AI Assistant for your documents.")

    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("1. Upload Your PDF")
        pdf_docs = st.file_uploader(
            "Upload your PDF here",
            type=["pdf"],
            accept_multiple_files=False
        )

        if pdf_docs:
            with st.spinner("Processing..."):
                # Use a temporary file to save the uploaded PDF
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(pdf_docs.read())
                    temp_file_path = tmp_file.name

                # Load and split documents
                loader = PyPDFLoader(temp_file_path)
                documents = loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200))
                os.remove(temp_file_path)
                
                # Create and store the conversation chain using the cached retriever
                retriever = get_hybrid_retriever(documents)
                st.session_state.conversation = get_conversation_chain(retriever)
                st.session_state.chat_history = []
            st.success("PDF processed successfully! You can now ask questions.")

    # Main chat interface
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

if __name__ == '__main__':
    main()