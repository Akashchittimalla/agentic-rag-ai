try:
    import dotenv
    import docling
    import langchain_community
    import chromadb
    import crewai
    import langgraph
    print("✅ EVERYTHING IS INSTALLED! You are ready to run ingest.py")
except ImportError as e:
    print(f"❌ STILL MISSING: {e.name}")
    print(f"Try running: pip install {e.name}")