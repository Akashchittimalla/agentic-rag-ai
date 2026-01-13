import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

def build_fast_database(pdf_path):
    print(f"🚀 Starting Fast Ingestion for: {pdf_path}")
    
    # 1. Clear old database to avoid mixing data
    if os.path.exists("./db"):
        print("🧹 Clearing old database...")
        shutil.rmtree("./db")

    # 2. Load PDF (Lightweight & Fast)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # 3. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    # 4. Create Local Embeddings
    print("📦 Creating new local database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 5. Save to local DB
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./db"
    )
    print(f"✅ Success! Database created for {pdf_path}")

if __name__ == "__main__":
    # Ensure this matches your file name in the folder
    target_file = "test1.pdf" 
    
    if os.path.exists(target_file):
        build_fast_database(target_file)
    else:
        print(f"❌ Error: {target_file} not found!")