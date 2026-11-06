from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

def get_embeddings():
    """Initialise et retourne les embeddings locaux Hugging Face."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm(model_name="phi3"):
    """Initialise et retourne le modèle local Ollama (par défaut phi3 pour éviter les surcharges RAM)."""
    return ChatOllama(model=model_name, temperature=0)

# Dans config.py :
def get_llm(model_name="phi3"):
    return ChatOllama(
        model=model_name,
        temperature=0,
        num_ctx=2048  # Limite la taille de la fenêtre de contexte pour éviter l'explosion de la VRAM/RAM
    )