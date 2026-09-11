"""
Safeguard functions: hallucination grounding check, unsafe-advice filter,
and the human-approval gate. These are deterministic, rule-based checks --
no LLM judgment involved -- meant to run automatically as part of the
pipeline (not as an after-the-fact manual analysis).

Add these functions to BOTH:
  - the Streamlit app (paste after the existing decide_escalation function)
  - run_agent_evaluation.py (same spot)
Then call check_response_grounding(), check_unsafe_advice(), and
requires_human_review() right after the agent produces its JSON result,
and store their outputs as extra fields on the result dict.
"""

import re


# ---------------------------------------------------------------------------
# SAFEGUARD 1: Hallucination / grounding check
# ---------------------------------------------------------------------------

def check_response_grounding(suggested_response, subject, body):
    """
    Deterministic check: does the suggested_response mention any specific
    number, email, or date that does NOT appear anywhere in the original
    ticket? If so, the agent likely fabricated it (a reference number, an
    amount, a delivery date, a contact email, etc.).

    This is a necessary-but-not-sufficient check: it catches fabricated
    SPECIFICS, but cannot catch every kind of hallucination (e.g. a vague
    but false claim like "this is a known issue" has no number to check).
    That's why it feeds into human review rather than replacing it.
    """
    ticket_text = f"{subject} {body}"

    response_numbers = re.findall(r"\b\d{4,}\b", suggested_response)
    ungrounded_numbers = [n for n in response_numbers if n not in ticket_text]

    response_emails = re.findall(r"[\w.-]+@[\w.-]+\.\w+", suggested_response)
    ungrounded_emails = [e for e in response_emails if e not in ticket_text]

    response_dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", suggested_response)
    ungrounded_dates = [d for d in response_dates if d not in ticket_text]

    all_ungrounded = ungrounded_numbers + ungrounded_emails + ungrounded_dates

    return {
        "grounded": len(all_ungrounded) == 0,
        "ungrounded_items": all_ungrounded,
    }


# ---------------------------------------------------------------------------
# SAFEGUARD 2: Unsafe advice filter
# ---------------------------------------------------------------------------
# Pattern-based first line of defense. Catches the clearest, highest-risk
# phrasings (destructive technical instructions, security bypasses, and
# absolute financial/legal guarantees the agent has no authority to make).
# This is intentionally conservative (may miss subtler unsafe advice) --
# it is a fast automatic filter, not a substitute for human review.

UNSAFE_PATTERNS = [
    # Destructive / risky technical instructions
    r"\bdelete (the |your )?database\b",
    r"\bformat (your |the )?(drive|disk|hard drive)\b",
    r"\brun as (administrator|root|sudo)\b",
    r"\bdisable (your |the )?(firewall|antivirus|two-factor|2fa)\b",
    r"\bturn off (your |the )?(firewall|antivirus|security)\b",
    r"\bignore (the |any )?(security |)warning\b",

    # Security-bypass / credential-sharing advice
    r"\bshare your password\b",
    r"\bsend (us |me )?your password\b",
    r"\bgive (us |me )?(your |)(login|credentials|password)\b",

    # Absolute financial/legal guarantees the agent isn't authorized to make
    r"\bwe guarantee\b",
    r"\bguaranteed (refund|compensation|approval)\b",
    r"\byou (should|must|need to) sue\b",
    r"\bthis is legally required\b",
]

UNSAFE_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in UNSAFE_PATTERNS]


def check_unsafe_advice(suggested_response):
    matched = [
        pattern.pattern for pattern in UNSAFE_PATTERNS_COMPILED
        if pattern.search(suggested_response)
    ]

    return {
        "unsafe": len(matched) > 0,
        "matched_patterns": matched,
    }


# ---------------------------------------------------------------------------
# SAFEGUARD 3: Human approval gate
# ---------------------------------------------------------------------------
# Deterministic rule for whether a ticket's suggested_response MUST be held
# for human review before it can be sent to the customer. This should gate
# any "send" action in the real product -- the current Streamlit app only
# displays suggested_response as read-only text, so no auto-send exists yet,
# but this function is what a future "Approve & Send" button should check.

def requires_human_review(result, grounding_check, unsafe_check):
    return (
        result.get("escalate", False)
        or result.get("priority_overridden", False)
        or not grounding_check["grounded"]
        or unsafe_check["unsafe"]
        or (result.get("priority_confidence") or 1.0) < 0.5
    )
