"""A small LangGraph RAG agent for local Markdown knowledge."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph


KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


class AgentState(TypedDict):
    question: str
    context: list[Document]
    answer: str


def load_knowledge() -> list[Document]:
    """Load and split Markdown files from the local knowledge directory."""
    documents: list[Document] = []
    for path in sorted(KNOWLEDGE_DIR.glob("**/*.md")):
        if path.name.startswith("_"):
            continue
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path.relative_to(Path(__file__).parent))},
            )
        )

    if not documents:
        raise RuntimeError(f"No Markdown documents found in {KNOWLEDGE_DIR}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    return splitter.split_documents(documents)


def build_graph():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Set it in the local .env file.")

    chunks = load_knowledge()
    retriever = BM25Retriever.from_documents(chunks, k=4)
    model = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful knowledge-base agent. Answer only from the provided context. "
                "If the context does not contain the answer, say you do not know.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    answer_chain = prompt | model | StrOutputParser()

    def retrieve(state: AgentState) -> dict[str, list[Document]]:
        return {"context": retriever.invoke(state["question"])}

    def generate(state: AgentState) -> dict[str, str]:
        context = "\n\n".join(
            f"[{document.metadata['source']}]\n{document.page_content}"
            for document in state["context"]
        )
        answer = answer_chain.invoke({"question": state["question"], "context": context})
        return {"answer": answer}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the local knowledge base.")
    parser.add_argument("question", nargs="+", help="Question to ask")
    args = parser.parse_args()

    graph = build_graph()
    result = graph.invoke({"question": " ".join(args.question), "context": [], "answer": ""})
    print(result["answer"])


if __name__ == "__main__":
    load_dotenv(Path(__file__).with_name(".env"))
    main()
