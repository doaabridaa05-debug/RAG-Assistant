import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

def cosine_similarity(vec1, vec2):
    """Calcule la similarité cosinus entre des vecteurs avec NumPy."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2.T)
    norm_vec1 = np.linalg.norm(vec1, axis=-1, keepdims=True)
    norm_vec2 = np.linalg.norm(vec2, axis=-1, keepdims=True)
    return dot_product / (norm_vec1 * norm_vec2.T + 1e-10)

def create_semantic_router(embeddings, llm):
    """
    Crée un routeur sémantique basé sur la similarité cosinus entre la question
    de l'utilisateur et des templates de prompts de référence.
    """
    physics_template = """You are a physics expert. Answer concisely:
{query}"""

    math_template = """You are a mathematics expert. Break down the problem step by step:
{query}"""

    general_template = """You are a helpful assistant. Answer the question:
{query}"""

    prompt_templates = [physics_template, math_template, general_template]

    # Vectoriser (embedder) les templates de prompts une seule fois
    prompt_embeddings = embeddings.embed_documents(prompt_templates)

    def prompt_router(input_data):
        query_text = input_data["query"]

        # Embed de la question utilisateur
        query_embedding = embeddings.embed_query(query_text)

        # Calcul de la similarité cosinus avec NumPy
        similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
        most_similar_template = prompt_templates[similarity.argmax()]

        if most_similar_template == physics_template:
            print("--> Domaine détecté : PHYSIQUE")
        elif most_similar_template == math_template:
            print("--> Domaine détecté : MATHÉMATIQUES")
        else:
            print("--> Domaine détecté : GÉNÉRAL")

        return PromptTemplate.from_template(most_similar_template)

    return RunnableLambda(prompt_router)