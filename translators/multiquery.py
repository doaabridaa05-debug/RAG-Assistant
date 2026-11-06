from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def create_multiquery_chain(llm):
    """Crée la chaîne LCEL pour générer plusieurs requêtes à partir d'une seule."""
    template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
    prompt = ChatPromptTemplate.from_template(template)

    return (
        prompt
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )

def get_unique_union(documents_list):
    """Fait l'union unique de tous les documents récupérés par les différentes requêtes."""
    flattened_docs = [doc for sublist in documents_list for doc in sublist]
    unique_docs = {doc.page_content: doc for doc in flattened_docs}
    return list(unique_docs.values())