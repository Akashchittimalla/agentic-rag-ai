import os
from typing import List, TypedDict
from dotenv import load_dotenv

# Core Frameworks
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage

load_dotenv()

# --- 1. SETUP THE BRAIN ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# --- 2. DEFINE THE STATE ---
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str

# --- 3. THE "RESEARCHER" NODE ---
def retrieve_docs(state: AgentState):
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    return {"documents": [context]}

# --- 4. THE "WRITER" NODE ---
def generate_response(state: AgentState):
    context = state["documents"][0]
    question = state["question"]
    
    prompt = f"""
    You are a helpful AI Assistant for Akash Chittimalla. 
    Use the provided graduation context to answer the question.
    
    CONTEXT:
    {context}
    
    QUESTION: 
    {question}
    
    INSTRUCTION: Be concise and professional. If the answer isn't in the context, say so.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"generation": response.content}

# --- 5. BUILD THE GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate", generate_response)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()

# --- 6. INTERACTIVE CHAT LOOP ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎓 GRADUATION ASSISTANT READY")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50)

    while True:
        user_input = input("\n🤔 Your Question: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye! Congratulations again on your graduation!")
            break
            
        inputs = {"question": user_input}
        print("🔍 Thinking...")
        result = app.invoke(inputs)
        
        print("\n🤖 Assistant:", result["generation"])