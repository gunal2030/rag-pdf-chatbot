import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_cohere import ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
# The following imports are for the new LCEL chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere.rerank import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

# Helper function to get or create an event loop to fix the async runtime error
def get_or_create_eventloop():
    """
    Retrieves the current event loop or creates a new one if it doesn't exist.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Caching functions to speed up processing
@st.cache_resource
def get_pdf_text(pdf_docs):
    """Extracts text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@st.cache_resource
def get_text_chunks(text):
    """Splits a long string of text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vector_store(text_chunks):
    """
    Creates and returns a FAISS vector store from text chunks.
    Ensures an event loop exists for embeddings initialization.
    """
    get_or_create_eventloop()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

@st.cache_resource
def get_rag_chain():
    """
    Builds and returns the complete RAG (Retrieval-Augmented Generation) chain.
    This replaces the deprecated `load_qa_chain`.
    """
    # Define the LLM model and prompt template
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt_template = """
    Answer the user's question from the provided context only.
    Context:\n {context}\n
    Question: {question}\n

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # This chain handles combining the retrieved documents with the prompt
    document_chain = create_stuff_documents_chain(llm, prompt)

    return document_chain

# Main application logic
def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("Chat with PDF :books:")
    st.write("Upload a PDF file to get started!")

    with st.sidebar:
        st.subheader("Your PDF files")
        pdf_docs = st.file_uploader(
            "Upload your PDF files here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    
                    # Create retriever and reranker
                    st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    cohere_rerank = CohereRerank(model="rerank-english-v3.0", top_n=3)
                    st.session_state.retriever = ContextualCompressionRetriever(
                        base_retriever=st.session_state.retriever,
                        base_compressor=cohere_rerank,
                    )
                    
                    # Store the new LCEL-based RAG chain in session state
                    st.session_state.chain = get_rag_chain()
                    
                    st.success("Processing complete!")
            else:
                st.error("Please upload a PDF file first!")

    if user_question := st.chat_input("Ask a question about your PDF:"):
        # Check if chain and retriever exist
        if "chain" not in st.session_state or "retriever" not in st.session_state:
            st.error("Please upload and process a PDF file first!")
            return
            
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # First retrieve the relevant documents
                    retrieved_docs = st.session_state.retriever.get_relevant_documents(user_question)
                    
                    # Then invoke the chain with the retrieved documents and question
                    response_text = st.session_state.chain.invoke({
                        "context": retrieved_docs, 
                        "question": user_question
                    })
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    return
                
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
if __name__ == "__main__":
    main()
