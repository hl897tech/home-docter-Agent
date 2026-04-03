import os
import logging
from typing import List, Tuple, Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from tools import triage  # 你的规则分流工具（假设返回 {"level": "...", "reasons": [...], "action": "..."}）

logger = logging.getLogger(__name__)
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ✅ English risk levels
RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class DoctorTriageResponse(BaseModel):
    risk_level: RiskLevel = Field(..., alias="risk_level")
    follow_up_questions: List[str] = Field(..., alias="follow_up_questions")
    possible_causes: List[str] = Field(..., alias="possible_causes")
    actions: List[str] = Field(..., alias="actions")
    emergency_signs: List[str] = Field(..., alias="emergency_signs")
    disclaimer: str = Field(..., alias="disclaimer")

    class Config:
        populate_by_name = True  # allow access by field name


def format_history(history: List[Tuple[str, str]]) -> str:
    """Compress chat history into plain text context (English)."""
    lines = []
    for role, content in history:
        prefix = "User" if role == "user" else "Assistant"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def build_main_executor():
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)
    tools = []  # later: tools = [triage, retrieve_docs]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a Family Doctor Q&A assistant for health education and triage guidance. "
            "You must NOT provide a definitive diagnosis and you do NOT replace a clinician.\n\n"
            "Your response MUST include ALL fields below:\n"
            "1) risk_level: one of [LOW, MEDIUM, HIGH]\n"
            "2) follow_up_questions: up to 3 items; if none, output ['None']\n"
            "3) possible_causes: common/possible causes (NOT a diagnosis; use cautious wording)\n"
            "4) actions: a clear action checklist\n"
            "5) emergency_signs: red flags / thresholds when urgent care is needed\n"
            "6) disclaimer: 1-2 sentences\n\n"
            "Safety rule: If the user describes chest pain, trouble breathing, altered consciousness, "
            "stroke signs, severe allergic reaction, heavy bleeding, or self-harm, you MUST advise "
            "seeking emergency care immediately (US: call 911).\n\n"
            "Output MUST be a JSON object with EXACT English keys:\n"
            "{\n"
            '  "risk_level": "...",\n'
            '  "follow_up_questions": [...],\n'
            '  "possible_causes": [...],\n'
            '  "actions": [...],\n'
            '  "emergency_signs": [...],\n'
            '  "disclaimer": "..."\n'
            "}\n"
            "Do NOT output any text outside the JSON."
        )),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False), llm


def run_pipeline(user_text: str, history: Optional[List[Tuple[str, str]]] = None) -> dict:
    """
    Pipeline:
    1) Rule-based triage (safety layer)
    2) If HIGH -> immediate emergency JSON response
    3) Else -> main agent (optionally with session history)
       Then do a second-pass structured output to ensure strict JSON schema
    """
    history = history or []

    # ✅ 1) safety triage first
    triage_struct = triage.invoke({"text": user_text})

    # NOTE: keep this consistent with your triage tool output
    if triage_struct.get("level") == "HIGH":
        reasons = triage_struct.get("reasons", [])
        action = triage_struct.get("action", "Seek emergency care now (US: call 911).")

        emergency_payload = {
            "risk_level": "HIGH",
            "follow_up_questions": ["None"],
            "possible_causes": reasons if reasons else ["Possible emergency red flags detected."],
            "actions": [action],
            "emergency_signs": ["Now"],
            "disclaimer": (
                "I cannot diagnose you online. This is safety guidance only; "
                "please follow emergency services/clinician evaluation."
            ),
        }
        return {"reply": emergency_payload, "meta": {"triage": triage_struct}}

    # ✅ 2) non-HIGH -> main agent
    executor, llm = build_main_executor()

    hist_text = format_history(history)
    combined_input = (
        f"[Chat History]\n{hist_text}\n\n[User Message]\n{user_text}"
        if hist_text else user_text
    )

    out = executor.invoke({"input": combined_input})
    raw = out.get("output", "")

    # ✅ 3) second pass: enforce strict structured JSON
    structured_llm = llm.with_structured_output(DoctorTriageResponse)
    try:
        result_obj = structured_llm.invoke(
            "Convert the content below into STRICT JSON with EXACT keys:\n"
            '"risk_level", "follow_up_questions", "possible_causes", "actions", '
            '"emergency_signs", "disclaimer".\n'
            "Rules:\n"
            "- follow_up_questions: max 3; if none use ['None']\n"
            "- All list fields must be JSON arrays\n"
            "- No extra keys\n\n"
            f"Content:\n{raw}"
        )
        reply_payload = result_obj.model_dump(by_alias=True)
    except (ValidationError, ValueError) as e:
        logger.warning("Structured output failed, fallback to raw output: %s", e)
        reply_payload = raw  # fallback (may not be strict JSON)

    return {"reply": reply_payload, "meta": {"triage": triage_struct}}