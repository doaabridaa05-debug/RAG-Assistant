from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def create_hyde_chain(llm):
    """Crée la chaîne LCEL pour générer un passage de document hypothétique."""
    template = """Please write a scientific paper passage to answer the question.
Question: {question}
Passage:"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()