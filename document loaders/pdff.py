import os
from dotenv import load_dotenv
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Load API key
load_dotenv()
api_key = os.getenv("MISTRAL_API_TOKEN")

# Streamlit UI
st.title("📄 PDF Q&A Chatbot")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create embeddings + vector store
    embeddings = MistralAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # Chat model
    model = ChatMistralAI(model="mistral-small-2402", api_key=api_key)

    # User question
    question = st.text_input("Ask a question about the PDF:")
    if question:
        # Retrieve relevant chunks
        relevant_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Prompt
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers based on the PDF."),
            ("user", "Context:\n{context}\n\nQuestion: {question}")
        ])
        prompt = template.format_messages(context=context, question=question)

        # Get answer
        response = model.invoke(prompt)
        st.write("### Answer:")
        st.write(response.content)
