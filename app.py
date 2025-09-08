import streamlit as st
import os
import io
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_cohere import ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_cohere.rerank import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
import asyncio

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

# Helper function to get or create an event loop
def get_or_create_eventloop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# Caching for the model to speed up processing
@st.cache_resource
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@st.cache_resource
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource
def get_vector_store(text_chunks):
    # Ensure event loop exists for embeddings initialization
    get_or_create_eventloop()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

@st.cache_resource
def get_conversational_chain():
    prompt_template = """
    Answer the user's question from the provided context only.
    Context:\n {context}?\n
    Question: {question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# New function to run async code in a synchronous context
def get_answer(user_question):
    # Ensure event loop exists for ainvoke call
    loop = get_or_create_eventloop()
    response = loop.run_until_complete(st.session_state.chain.ainvoke({"input_documents": st.session_state.retriever.get_relevant_documents(user_question), "question": user_question}))
    return response["output_text"]

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
                    # FIX: Reduce k to prevent ResourceExhausted error
                    st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    cohere_rerank = CohereRerank(model="rerank-english-v3.0", top_n=3)
                    st.session_state.retriever = ContextualCompressionRetriever(
                        base_retriever=st.session_state.retriever,
                        base_compressor=cohere_rerank,
                    )
                    st.session_state.chain = get_conversational_chain()
                    st.success("Processing complete!")
            else:
                st.error("Please upload a PDF file first!")

    if user_question := st.chat_input("Ask a question about your PDF:"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_answer(user_question)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
