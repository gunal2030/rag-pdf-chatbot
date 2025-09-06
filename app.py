import streamlit as st
import os
import io
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_cohere.rerank import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]

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
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    Do not add any additional information outside of the provided context. If the answer is not in the provided context,
    just say, "Answer is not available in the context."\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def get_answer(user_question):
    with st.spinner("Processing..."):
        docs = st.session_state.retriever.get_relevant_documents(user_question)
        response = st.session_state.chain.invoke({"input_documents": docs, "question": user_question})
        return response["output_text"]

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📚")
    st.header("Chat with your PDF :books:")

    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.sidebar:
        st.title("Settings")
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs and click Process", type="pdf", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 20})
                    cohere_rerank = CohereRerank(top_n=3)
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