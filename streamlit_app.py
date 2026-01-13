import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

load_dotenv()

# Page Config
st.set_page_config(page_title="AI Document Assistant", page_icon="🤖")
st.title("🤖 Artificial Intelligence Document Assistant")
st.markdown("Ask questions about your uploaded PDF documents.")

# Initialize Brain (Groq)
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=os.getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Connect to Database
if os.path.exists("./db"):
    vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()

    # User Input
    user_query = st.text_input("Enter your question about the PDF:")

    if user_query:
        with st.spinner("Artificial Intelligence is thinking..."):
            # 1. Retrieve
            docs = retriever.invoke(user_query)
            context = "\n".join([d.page_content for d in docs])
            
            # 2. Generate
            prompt = f"Use this context to answer: {context}\n\nQuestion: {user_query}"
            response = llm.invoke([HumanMessage(content=prompt)])
            
            st.subheader("Assistant Response:")
            st.write(response.content)
else:
    st.error("Database not found. Please run ingest.py first!")