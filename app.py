import streamlit as st
import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage

import config
from compressors.reranker import create_rerank_retriever
from correctors.crag import get_crag_grader
from memory.conversation import get_history_aware_retriever

st.set_page_config(page_title="RAG Local Modulaire & CRAG", layout="wide")

st.title("🤖 Assistant RAG Local, Modulaire & Interactif (CRAG + Cohere + Phi3)")
st.markdown("Interface utilisateur Streamlit connectée à votre pipeline RAG local 100% sans frais OpenAI.")

@st.cache_resource
def init_rag_system():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = config.get_embeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

vectorstore = init_rag_system()
llm = config.get_llm("phi3")
history_retriever = get_history_aware_retriever(vectorstore, model_name="phi3")
rerank_retriever = create_rerank_retriever(vectorstore, k_initial=10, top_n=2)
grader = get_crag_grader(model_name="phi3")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

user_query = st.chat_input("Posez votre question sur les agents LLM...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Recherche, Re-ranking Cohere, et Évaluation CRAG en cours..."):
        try:
            # 1. Récupération et Re-ranking
            ranked_docs = rerank_retriever.invoke(user_query)

            # 2. CRAG Validation
            valid_docs = []
            for doc in ranked_docs:
                try:
                    score = grader.invoke({"question": user_query, "context": doc.page_content}).binary_score
                except Exception:
                    score = "yes"
                if score == "yes":
                    valid_docs.append(doc)

            if not valid_docs:
                valid_docs = ranked_docs[:1] # Fallback

            context_text = "\n\n".join([doc.page_content for doc in valid_docs])

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.output_parsers import StrOutputParser

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "Répondez à la question de manière claire en vous basant uniquement sur le contexte validé suivant :\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            chain = qa_prompt | llm | StrOutputParser()
            response_text = chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history,
                "context": context_text
            })

            with st.chat_message("assistant"):
                st.markdown(response_text)

            st.session_state.chat_history.extend([
                HumanMessage(content=user_query),
                AIMessage(content=response_text)
            ])
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")