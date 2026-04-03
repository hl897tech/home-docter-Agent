from langchain_core.tools import tool
from triage import triage_rule

@tool
def triage(text: str) -> dict:
    """Rule-based triage. Detect red flags and return risk level and actions."""
    return triage_rule(text)