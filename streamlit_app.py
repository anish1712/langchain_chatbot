"""Local browser UI for the LangGraph RAG agent."""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from app import build_graph, load_knowledge


PROJECT_DIR = Path(__file__).parent
load_dotenv(PROJECT_DIR / ".env")

st.set_page_config(
    page_title="Atlas support desk",
    page_icon=":material/support_agent:",
    layout="centered",
    initial_sidebar_state="expanded",
)

SUGGESTIONS = [
    "How long do refunds take?",
    "What are the support hours?",
    "How long does express shipping take?",
]


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_graph()


@st.cache_data(show_spinner=False)
def get_knowledge_stats() -> tuple[int, list[str]]:
    chunks = load_knowledge()
    sources = sorted({document.metadata["source"] for document in chunks})
    return len(chunks), sources


def reset_chat() -> None:
    st.session_state.messages = []


def clean_answer(text: str) -> str:
    """Keep source metadata out of the visible conversation."""
    return re.sub(r"\s*Sources:\s*[^\n]+", "", text, flags=re.IGNORECASE).strip()


def ask_agent(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/support_agent:"):
        status = st.status("Searching the knowledge base", expanded=False)
        try:
            with status:
                result = get_graph().invoke(
                    {"question": question, "context": [], "answer": ""}
                )
                answer = clean_answer(result["answer"])
            status.update(label="Answer ready", state="complete")
        except RuntimeError as error:
            status.update(label="Configuration needed", state="error")
            st.error(str(error))
        except Exception as error:
            status.update(label="Request failed", state="error")
            st.error(f"The agent could not answer this question: {error}")
        else:
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )


with st.sidebar:
    st.markdown("## :material/support_agent: Atlas")
    st.caption("Grounded answers for your support team")
    st.space("small")

    try:
        chunk_count, sources = get_knowledge_stats()
        st.badge("Agent online", icon=":material/check_circle:", color="green")
        st.metric("Knowledge chunks", chunk_count)
        with st.expander("Knowledge sources", icon=":material/folder_open:"):
            for source in sources:
                st.caption(source)
    except RuntimeError as error:
        st.badge("Needs setup", icon=":material/warning:", color="orange")
        st.caption(str(error))

    st.space("medium")
    if st.button("Start a new chat", icon=":material/add:", width="stretch"):
        reset_chat()
        st.rerun()

st.title("Support desk", anchor=False)
st.write("Ask a question and get an answer grounded in your local knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = ":material/person:" if message["role"] == "user" else ":material/support_agent:"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(clean_answer(message["content"]))

if not st.session_state.messages:
    st.subheader("Try a question", anchor=False)
    selected = st.pills(
        "Suggested questions",
        SUGGESTIONS,
        label_visibility="collapsed",
    )
    if selected:
        ask_agent(selected)

prompt = st.chat_input("Ask about refunds, shipping, or support hours")
if prompt:
    ask_agent(prompt)
