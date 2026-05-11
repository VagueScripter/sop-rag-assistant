import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
KNOWLEDGE_BASE_DIR = "knowledge_bases"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
