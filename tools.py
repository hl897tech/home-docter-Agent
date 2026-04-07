from langchain_core.tools import tool
from triage import triage_rule


@tool
def triage(text: str) -> dict:
    """Rule-based triage. Detect red flags and return risk level and actions."""
    return triage_rule(text)


@tool
def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge base for information about symptoms, diseases, drugs, or first aid.
    Use this tool whenever the user describes symptoms or asks about a medical condition."""
    from retriever import get_retriever
    retriever = get_retriever(k=3)
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant medical information found in the knowledge base."
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )