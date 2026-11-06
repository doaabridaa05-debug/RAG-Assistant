import datetime
from typing import Optional
from pydantic import BaseModel, Field  # <-- Import corrigé ici
from langchain_core.prompts import ChatPromptTemplate

class TutorialSearch(BaseModel):
    """Recherche dans une base de données de vidéos de tutoriels sur une bibliothèque logicielle."""

    content_search: str = Field(
        ...,
        description="Requête de recherche sémantique appliquée aux transcripts des vidéos.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Version abrégée de la recherche pour le titre de la vidéo. "
            "Doit être succincte et ne contenir que les mots-clés présents dans un titre."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Filtre de nombre de vues minimum, inclusif.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Filtre de nombre de vues maximum, exclusif.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Filtre de date de publication la plus ancienne, inclusive.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Filtre de date de publication la plus récente, exclusive.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Longueur minimale de la vidéo en secondes, inclusive.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Longueur maximale de la vidéo en secondes, exclusive.",
    )

    def pretty_print(self) -> None:
        """Affiche joliment les filtres extraits."""
        for field_name, field_info in self.model_fields.items():
            val = getattr(self, field_name)
            if val is not None:
                print(f"{field_name}: {val}")

def create_query_analyzer_chain(llm):
    """Crée la chaîne LCEL pour transformer une question en langage naturel en requête structurée."""
    system = """Vous êtes un expert pour convertir des questions d'utilisateurs en requêtes de base de données. \
Vous avez accès à une base de données de tutoriels vidéo pour construire des applications avec des LLM. \
Étant donné une question, retournez une requête de base de données optimisée."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{question}"),
    ])

    structured_llm = llm.with_structured_output(TutorialSearch)
    return prompt | structured_llm