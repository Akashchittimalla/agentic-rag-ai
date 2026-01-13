import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Agentic RAG", page_icon="🧠")
st.title("🧠 Agentic RAG Document Assistant")
st.markdown("Upload a PDF and ask the AI questions about its content.")

# --- SIDEBAR: API KEY & CONFIG ---
# This pulls from your Streamlit Secrets
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in Secrets! please check Advanced Settings.")
    st.stop()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # 1. Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    # 2. Ingestion Process
    with st.spinner("Analyzing document..."):
        # Load and Split
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create temporary VectorStore in memory/temp folder
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever()

    st.info("System Ready. Ask your question below.")

    # 3. Chat Interface
    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("AI is thinking..."):
            # Initialize LLM
            llm = ChatGroq(
                model="llama-3.3-70b-versatile", 
                groq_api_key=groq_api_key
            )
            
            # Retrieve Context
            relevant_docs = retriever.invoke(user_query)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate Answer
            prompt = f"""
            You are a helpful AI assistant. Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know.
            
            Context: {context}
            
            Question: {user_query}
            """
            
            response = llm.invoke([HumanMessage(content=prompt)])
            
            st.subheader("Answer:")
            st.write(response.content)

else:
    st.warning("Please upload a PDF to get started.")
