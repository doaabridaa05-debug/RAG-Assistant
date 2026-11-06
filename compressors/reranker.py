
import os
import cohere

# Définition explicite de la clé si ce n'est pas fait dans config.py
os.environ["COHERE_API_KEY"] = "YOUR_API_KEY"

class CustomCohereReranker:
    def __init__(self, vectorstore, k_initial=10, top_n=3):
        self.vectorstore = vectorstore
        self.k_initial = k_initial
        self.top_n = top_n
        # Instanciation sécurisée en passant la clé directement si besoin
        self.client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))

    def invoke(self, question: str):
        """
        1. Récupère k_initial documents via la recherche sémantique (Chroma).
        2. Ré-ordonne ces documents via l'API Cohere Rerank.
        3. Retourne les top_n documents les plus pertinents.
        """
        # Étape 1 : Recherche large
        docs = self.vectorstore.similarity_search(question, k=self.k_initial)
        if not docs:
            return []

        # Extraction des textes
        doc_contents = [doc.page_content for doc in docs]

        # Étape 2 : Appel au modèle de ré-ordonnancement Cohere
        response = self.client.rerank(
            model="rerank-english-v3.0",
            query=question,
            documents=doc_contents,
            top_n=self.top_n
        )

        # Étape 3 : Reconstruction de la liste ordonnée des documents
        reranked_docs = []
        for result in response.results:
            original_doc = docs[result.index]
            original_doc.metadata["relevance_score"] = result.relevance_score
            reranked_docs.append(original_doc)

        return reranked_docs

def create_rerank_retriever(vectorstore, k_initial=10, top_n=3):
    """Fonction utilitaire pour s'aligner avec votre architecture modulaire."""
    return CustomCohereReranker(vectorstore, k_initial=k_initial, top_n=top_n)
