# LangChain + LangGraph RAG Agent

A minimal question-answering agent that retrieves relevant Markdown from `knowledge/` with local BM25 search before calling a Groq chat model.

## Setup

Create a virtual environment and install dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY` to a newly generated key. Do not commit `.env` or share the key.

## Run

```powershell
py app.py "How long do refunds take?"
```

## Run the browser frontend

```powershell
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser. The frontend keeps chat history for the current browser session and uses the same LangGraph agent as the CLI.

Add or edit Markdown files under `knowledge/`; they are loaded and indexed when the app starts. Retrieved source files remain available in the frontend sidebar but are not appended to chat answers.

## Flow

`question -> retrieve top 4 chunks -> generate grounded answer`

The graph is defined in `app.py` with LangGraph nodes named `retrieve` and `generate`. Retrieval uses local BM25 ranking, so it does not consume embedding API credits. Override the chat model with `GROQ_MODEL`.
