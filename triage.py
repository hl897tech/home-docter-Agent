import re
from typing import Dict, Any

# Emergency red flag patterns (English)
RED_FLAG_PATTERNS = [
    (r"(chest pain|chest tightness|pressure in chest)",
     "Possible cardiopulmonary emergency risk"),

    (r"(shortness of breath|difficulty breathing|can't breathe|breathless)",
     "Shortness of breath is an emergency warning sign"),

    (r"(confusion|unconscious|seizure|fainted|passed out)",
     "Altered consciousness or seizure is an emergency sign"),

    (r"(slurred speech|face drooping|one-sided weakness|paralysis)",
     "Possible stroke symptoms"),

    (r"(severe allergic reaction|throat swelling|anaphylaxis)",
     "Possible severe allergic reaction"),

    (r"(heavy bleeding|coughing blood|vomiting blood|black stool)",
     "Possible serious bleeding"),

    (r"(sudden severe headache|worst headache of my life)",
     "Possible acute cerebrovascular event"),

    (r"(suicidal|want to kill myself|don't want to live)",
     "Mental health emergency risk"),
]


def triage_rule(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    hits = []

    for pattern, reason in RED_FLAG_PATTERNS:
        if re.search(pattern, text_lower):
            hits.append(reason)

    if hits:
        return {
            "level": "HIGH",
            "reasons": hits,
            "action": (
                "Possible emergency symptoms detected. "
                "Seek emergency care immediately or call your local emergency number "
                "(US: call 911). Do not rely on online consultation."
            )
        }

    return {
        "level": "LOW",
        "reasons": [],
        "action": (
            "No obvious emergency red flags detected. "
            "You may proceed with further assessment."
        )
    }