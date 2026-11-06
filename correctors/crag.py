from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class GradeDocuments(BaseModel):
    """Évaluation de la pertinence binaire du document."""
    binary_score: Literal["yes", "no"] = Field(
        description="Le document est-il pertinent pour répondre à la question ? 'yes' ou 'no'"
    )

def get_crag_grader(model_name="phi3"):
    llm = ChatOllama(model=model_name, temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system_prompt = """Vous êtes un examinateur rigoureux qui évalue la pertinence d'un document récupéré par rapport à une question utilisateur.
Si le document contient des informations ou des concepts utiles pour répondre à la question, répondez 'yes'. Sinon, répondez 'no'."""

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question de l'utilisateur : {question}\n\nDocument récupéré : {context}\n\nVotre évaluation (yes/no) :")
    ])

    return grade_prompt | structured_llm_grader
