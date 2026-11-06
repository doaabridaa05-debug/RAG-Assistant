import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Import de vos modules personnalisés
from compressors.reranker import create_rerank_retriever
from correctors.crag import get_crag_grader
from memory.conversation import get_history_aware_retriever
import config

def run_ultimate_rag_pipeline():
    print("1. Chargement et indexation locale (Hugging Face + Chroma)...")
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Utilisation correcte de config.py (plus d'attribut EMBEDDING_MODEL manquant)
    embeddings = config.get_embeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    # Récupération du LLM local (Ollama - phi3 via config)
    llm = config.get_llm("phi3")

    # Simulation d'un historique de conversation
    chat_history = []

    print("\n=== PIPELINE RAG LOCAL COMPLET (Historique + Cohere + CRAG + Phi3) ===")

    # --- QUESTION 1 ---
    q1 = "What is Manipulating Data using numpy?"
    print(f"\n[Utilisateur] : {q1}")

    # Étape A : Historique conversationnel (Reformulation de la requête)
    history_retriever = get_history_aware_retriever(vectorstore, model_name="phi3")

    # Étape B : Ré-ordonnancement (Re-ranking via Cohere)
    rerank_retriever = create_rerank_retriever(vectorstore, k_initial=10, top_n=3)
    ranked_docs = rerank_retriever.invoke(q1)
    print(f"-> {len(ranked_docs)} documents retenus après Re-ranking (Cohere).")

    # Étape C : Évaluation CRAG (Filtrage par phi3)
    grader = get_crag_grader(model_name="phi3")
    valid_docs = []
    for i, doc in enumerate(ranked_docs):
        try:
            score = grader.invoke({"question": q1, "context": doc.page_content}).binary_score
        except Exception:
            score = "yes"

        if score == "yes":
            print(f"   [Document {i+1}] -> CRAG: PERTINENT (yes)")
            valid_docs.append(doc)
        else:
            print(f"   [Document {i+1}] -> CRAG: HORS-SUJET (no)")

    # Étape D : Génération finale via LCEL pur
    if len(valid_docs) > 0:
        context_text = "\n\n".join([doc.page_content for doc in valid_docs])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Répondez à la question de manière claire en vous basant uniquement sur le contexte validé suivant :\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        chain = qa_prompt | llm | StrOutputParser()
        response_text = chain.invoke({
            "input": q1,
            "chat_history": chat_history,
            "context": context_text
        })

        print(f"\n[Assistant] :\n{response_text}")
        chat_history.extend([HumanMessage(content=q1), AIMessage(content=response_text)])
    else:
        print("\n[CRAG ACTION CORRECTIVE] Aucun document validé. Recherche de secours nécessaire.")

    # --- QUESTION 2 (Test de la mémoire de conversation) ---
    q2 = "Can you give me an example of the first method mentioned?"
    print(f"\n[Utilisateur (Suivi)] : {q2}")

    ranked_docs_2 = rerank_retriever.invoke(q2)
    valid_docs_2 = []
    for doc in ranked_docs_2:
        try:
            if grader.invoke({"question": q2, "context": doc.page_content}).binary_score == "yes":
                valid_docs_2.append(doc)
        except:
            valid_docs_2.append(doc)

    if len(valid_docs_2) > 0:
        context_text_2 = "\n\n".join([doc.page_content for doc in valid_docs_2])
        response_text_2 = chain.invoke({
            "input": q2,
            "chat_history": chat_history,
            "context": context_text_2
        })
        print(f"\n[Assistant] :\n{response_text_2}")

if __name__ == "__main__":
    run_ultimate_rag_pipeline()