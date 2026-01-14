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
st.set_page_config(page_title="AI Agentic RAG", page_icon="🧠", layout="wide")

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if st.button("🗑️ Clear Chat History"):
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### How it works")
    st.caption("1. PDF is chunked into vectors\n2. ChromaDB stores embeddings\n3. Llama 3.3 retrieves & answers")

# --- MAIN INTERFACE ---
st.title("🧠 Agentic RAG Document Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

groq_api_key = os.getenv("GROQ_API_KEY")

if uploaded_file is not None:
    # 1. Processing (using temp file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Check if we need to process the file (to avoid re-processing on every click)
    if "vectorstore" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("🔄 Building AI Knowledge Base..."):
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            st.session_state.last_file = uploaded_file.name
            st.success("✅ Document Analyzed!")

    # 2. Chat UI
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about your PDF:"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
                retriever = st.session_state.vectorstore.as_retriever()
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
                response = llm.invoke([HumanMessage(content=full_prompt)])
                
                st.markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
else:
    st.info("👈 Please upload a PDF in the sidebar to begin.")