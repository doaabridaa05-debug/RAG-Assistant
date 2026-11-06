# 🤖 Local Modular RAG Assistant (CRAG + Cohere + Phi-3 + Streamlit)

A robust, modular, and privacy-first Retrieval-Augmented Generation (RAG) pipeline running **100% locally** to avoid OpenAI costs and rate limits. This project integrates state-of-the-art advanced RAG techniques—including **Corrective RAG (CRAG)**, **Cohere Re-ranking**, **Conversational Memory**, and a clean **Streamlit Web UI**—all powered by lightweight local models.

---

## 🌟 Key Features & Benefits

* **Zero OpenAI Costs & Quotas:** Bypasses API rate limits (HTTP 429) by utilizing local embeddings (**Hugging Face** `all-MiniLM-L6-v2`) and local LLMs (**Ollama** `phi3`).
* **Advanced Re-ranking (Cohere):** Combines vector retrieval with **Cohere Re-ranking** (`rerank-english-v3.0`) to surface the most contextually relevant document chunks.
* **Corrective RAG (CRAG):** Features an automated binary relevance evaluator (`yes`/`no`) powered by `phi3` to filter out hallucinated or off-topic retrieved documents before generation.
* **Conversational Memory:** Uses a history-aware contextual retriever in pure LCEL to seamlessly handle multi-turn follow-up questions.
* **Modular Architecture:** Cleanly decoupled code structure ensuring high maintainability and ease of extension.
* **Interactive Web UI:** Comes with a ready-to-use **Streamlit** interface for natural chatting.

---

## 📂 Project Structure

```text
RAG/
│
├── compressors/                # Re-ranking module
│   └── reranker.py             # Custom Cohere Re-rank wrapper
├── correctors/                 # Corrective RAG (CRAG) module
│   └── crag.py                 # Binary relevance grader (Pydantic + Phi-3)
├── memory/                     # Conversation history module
│   └── conversation.py         # History-aware query transformer
├── translators/                # Query expansion/translation (optional/extensible)
├── routers/                    # Semantic/Logical routing (optional/extensible)
├── config.py                   # Global configuration & client loaders
├── MAIN.py                     # Command-line orchestration script
├── app.py                      # Streamlit Interactive Web Interface
└── README.md                   # Project Documentation
```
## 🛠️ Tech Stack

- Orchestration & Framework: LangChain (LCEL)

- Vector Database: Chroma (In-memory/Local persistence)

- Embeddings: Hugging Face (all-MiniLM-L6-v2)

- Local LLM Server: Ollama (phi3 optimized with context window management)

- Re-ranking API: Cohere SDK (rerank-english-v3.0)

- Frontend UI: Streamlit

  ## 🚀 Installation & Setup

  1. Clone the Repository
  ```code
  git clone [https://github.com/doaabridaa05-debug/RAG-Assistant/new/main](https://github.com/doaabridaa05-debug/RAG-Assistant/new/main)
  cd doaabridaa05-debug
  ```
2. Create and Activate Virtual Environment

```code
python -m venv .venv
# On Windows (PowerShell):
& '.\.venv\Scripts\Activate.ps1'

```
3. Install Dependencies
 ```code
pip install -r requirements.txt
```

5. Ensure Ollama is Running
Make sure you have Ollama installed, running locally, and the phi3 model pulled:

```code
ollama pull phi3

```
##⚙️ Configuration (reranker.py)

in your reranker file at the root directory to set up your API keys and local model factories:

```code
# Set your Cohere API Key
os.environ["COHERE_API_KEY"] = "YOUR_COHERE_API_KEY_HERE"
```
## 💻 Usage
Option A: Run via Command-Line Interface (MAIN.py)
```code
python MAIN.py
```
Option B: Run via Streamlit Web UI (app.py)
To launch the interactive chat interface in your browser:

```code

streamlit run app.py
```
##📄 License:

This project is open-source under the MIT License.



































