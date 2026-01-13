import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Check if DB exists
if not os.path.exists("./db"):
    st.info("Creating database from PDF... Please wait.")
    
    # Load your PDF (make sure test1.pdf is uploaded to GitHub!)
    loader = PyPDFLoader("test2.pdf")
    data = loader.load()
    
    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # Create Embeddings and Save to DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="./db")
    st.success("Database created successfully!")
