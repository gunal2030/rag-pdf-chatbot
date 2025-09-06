import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatCohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI # Changed this line
from langchain.prompts import PromptTemplate
from langchain_cohere.rerank import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set up environment variables
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
cohere_api_key = os.getenv("COHERE_API_KEY")

def get_answer(user_question):
    # Load the PDF content
    text = "Your PDF content as a single string goes here."

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_text(text)

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vectorstore = FAISS.from_texts(docs, embeddings)

    # Set up the Cohere reranker
    compressor = CohereRerank(cohere_api_key=cohere_api_key, top_n=3)

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Changed this line
    compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=compressor)
    
    # Set up the LLM and RAG chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=google_api_key) # Changed this line
    prompt_template = """
    You are a professional assistant for answering questions based on the provided context only.
    You must provide concise and relevant information from the given context.
    If the answer is not available in the context, politely state that the information is not in the document.

    Context:\n {context}\n
    Question: {question}

    Answer:
    """
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=PromptTemplate.from_template(prompt_template))

    # Get the answer
    retrieved_docs = compression_retriever.invoke(user_question)
    response = qa_chain.invoke({"input_documents": retrieved_docs, "question": user_question})

    return response

def main():
    st.title("RAG PDF Chatbot")
    user_question = st.text_input("Ask a question about the PDF:")
    if user_question:
        try:
            response = get_answer(user_question)
            st.write(response["output_text"])
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()