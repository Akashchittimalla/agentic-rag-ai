# 🧠 Agentic RAG Document Assistant

An end-to-end **Artificial Intelligence** platform that transforms static PDF documents into dynamic, searchable knowledge bases. This project implements a **Retrieval-Augmented Generation (RAG)** pipeline using the Llama 3.3 model to provide grounded, context-aware answers to complex queries.



## 🚀 Live Demo
**[INSERT YOUR STREAMLIT URL HERE]**

## ✨ Key Features
* **Dynamic Ingestion:** Upload any PDF and instantly build a searchable vector index.
* **Agentic Reasoning:** Powered by **Llama 3.3-70B** via Groq for high-speed, logical synthesis of retrieved data.
* **Vector Storage:** Utilizes **ChromaDB** for high-dimensional vector embeddings and efficient semantic search.
* **Automated Evaluation:** Includes an "LLM-as-a-Judge" framework to score Groundedness and Relevance.
* **Persistent Chat:** Streamlit session-state management for multi-turn conversations.

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Orchestration:** LangChain
* **LLM:** Llama 3.3-70B (Groq)
* **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
* **Vector DB:** ChromaDB
* **Frontend:** Streamlit

## 📊 Performance & Evaluation
To ensure reliability and prevent hallucinations, this project uses the **RAG Triad** evaluation framework.

| Metric | Score | Definition |
| :--- | :--- | :--- |
| **Groundedness** | **5/5** | Answers are derived strictly from the provided PDF context. |
| **Relevance** | **5/5** | Answers directly and accurately address the user's intent. |

> *Evaluated using an automated batch-testing suite with Llama 3.3 as the judge.*

## ⚙️ Installation & Local Setup

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/agentic-rag-ai.git](https://github.com/YOUR_USERNAME/agentic-rag-ai.git)
   cd agentic-rag-ai
