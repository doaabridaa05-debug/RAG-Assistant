from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def create_decomposition_chain(llm):
    """Crée la chaîne LCEL pour décomposer une question complexe en sous-questions."""
    template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
Decompose the following complex question into 3 distinct sub-questions that can be answered step-by-step: \n
Question: {question} \n
Output (3 sub-questions, one per line):"""
    prompt = ChatPromptTemplate.from_template(template)

    return (
        prompt
        | llm
        | StrOutputParser()
        | (lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )