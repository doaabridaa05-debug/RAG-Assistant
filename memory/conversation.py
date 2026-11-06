from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

def get_history_aware_retriever(vectorstore, model_name="phi3"):
    """
    Crée un retriever conscient de l'historique conversationnel via LCEL pur,
    évitant ainsi toute dépendance aux modules de chaînes obsolètes.
    """
    llm = ChatOllama(model=model_name, temperature=0)

    # Prompt de reformulation de la question avec l'historique
    contextualize_q_system_prompt = (
        "Étant donné un historique de conversation et la dernière question de l'utilisateur "
        "qui pourrait faire référence à des éléments du contexte précédent, "
        "formulez une question autonome qui puisse être comprise sans l'historique de la conversation. "
        "Ne répondez PAS à la question, contentez-vous de la reformuler si nécessaire ou renvoyez-la telle quelle."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Chaîne LCEL pour transformer la requête
    query_transformer_chain = contextualize_q_prompt | llm | StrOutputParser()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Wrapper personnalisé compatible avec la méthode .invoke({"input": ..., "chat_history": ...})
    class HistoryAwareRetrieverWrapper:
        def __init__(self, transformer, retriever):
            self.transformer = transformer
            self.retriever = retriever

        def invoke(self, input_dict):
            chat_history = input_dict.get("chat_history", [])
            user_input = input_dict.get("input", "")

            if chat_history:
                # Reformule la question en tenant compte de l'historique
                standalone_query = self.transformer.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
            else:
                standalone_query = user_input

            # Exécute la recherche vectorielle avec la requête (reformulée ou brute)
            return self.retriever.invoke(standalone_query)

    return HistoryAwareRetrieverWrapper(query_transformer_chain, base_retriever)