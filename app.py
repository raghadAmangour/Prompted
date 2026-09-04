"""
AI Support Ticket Triage — Streamlit App
ML (Issue Type + Priority) + Agentic GenAI layer (Groq-hosted Llama, tool calling)

Deploy notes:
- Put issue_type_model.joblib and priority_model.joblib inside the models/ folder
  next to this file (NOT an external path like ../Downloads/...).
- Set GROQ_API_KEY in Streamlit Cloud's "Secrets" panel (or a local .streamlit/secrets.toml):
    GROQ_API_KEY = "gsk_..."
"""

import json
import re

import joblib
import pandas as pd
import streamlit as st
from groq import Groq

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

ISSUE_TYPE_MODEL_PATH = "models/issue_type_model.joblib"
PRIORITY_MODEL_PATH = "models/priority_model.joblib"

LLAMA_MODEL = "openai/gpt-oss-20b"   # or "openai/gpt-oss-120b" for higher quality
# NOTE: llama-3.1-8b-instant and llama-3.3-70b-versatile were deprecated/shut down
# by Groq on 2026-08-16. Use the openai/gpt-oss-* models instead (or check
# https://console.groq.com/docs/models for the current list).
TEMPERATURE = 0.2
MAX_AGENT_STEPS = 6

ALLOWED_QUEUES = [
    "Billing and Payments",
    "Returns and Exchanges",
    "Technical Support",
    "Account Management",
    "General Inquiry",
]

REQUIRED_GENAI_FIELDS = [
    "predicted_queue",
    "summary",
    "main_problem",
    "recommended_action",
    "suggested_response",
]

# ---------------------------------------------------------------------------
# 2. CLIENT + MODELS (cached so they load once per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. Add it under Settings → Secrets "
            "in Streamlit Cloud (or .streamlit/secrets.toml locally)."
        )
        st.stop()
    return Groq(api_key=api_key)


@st.cache_resource
def load_ml_models():
    try:
        issue_type_model = joblib.load(ISSUE_TYPE_MODEL_PATH)
        priority_model = joblib.load(PRIORITY_MODEL_PATH)
        return issue_type_model, priority_model
    except FileNotFoundError as e:
        st.error(
            f"Could not find ML model files: {e}. "
            f"Make sure issue_type_model.joblib and priority_model.joblib "
            f"are inside the models/ folder in the repo."
        )
        st.stop()


client = get_groq_client()
issue_type_model, priority_model = load_ml_models()

# ---------------------------------------------------------------------------
# 3. ML PREDICTIONS
# ---------------------------------------------------------------------------


def predict_ticket_labels(subject, body):
    subject = "" if pd.isna(subject) else str(subject).strip()
    body = "" if pd.isna(body) else str(body).strip()
    ticket_text = f"{subject} {body}".strip()

    if not ticket_text:
        raise ValueError("Ticket text cannot be empty.")

    predicted_type = str(issue_type_model.predict([ticket_text])[0])
    predicted_priority = str(priority_model.predict([ticket_text])[0])

    return {
        "ticket_text": ticket_text,
        "predicted_type": predicted_type,
        "predicted_priority": predicted_priority,
    }


# ---------------------------------------------------------------------------
# 4. AGENT TOOLS (operate only on ticket text + our own ML models)
# ---------------------------------------------------------------------------


def get_ml_confidence(subject: str, body: str) -> dict:
    ticket_text = f"{subject} {body}".strip()
    result = {}
    for name, model in [("issue_type", issue_type_model), ("priority", priority_model)]:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([ticket_text])[0]
            classes = model.classes_
            best_idx = probs.argmax()
            result[name] = {
                "predicted_label": str(classes[best_idx]),
                "confidence": round(float(probs[best_idx]), 3),
            }
        else:
            result[name] = {
                "predicted_label": str(model.predict([ticket_text])[0]),
                "confidence": None,
            }
    return result


URGENCY_KEYWORDS = [
    "urgent", "asap", "immediately", "critical", "emergency",
    "down", "outage", "security breach", "unauthorized",
    "can't access", "cannot access", "data loss", "broken",
]


def analyze_urgency_signals(subject: str, body: str) -> dict:
    text = f"{subject} {body}".lower()
    matched = [kw for kw in URGENCY_KEYWORDS if kw in text]
    return {"urgency_score": len(matched), "matched_keywords": matched}


def extract_ticket_entities(subject: str, body: str) -> dict:
    text = f"{subject} {body}"
    reference_numbers = re.findall(r"\b(?:INV|ORD|REF)?-?\d{4,}\b", text)
    emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    return {"reference_numbers": reference_numbers, "emails": emails, "dates": dates}


def decide_escalation(priority: str, urgency_score: int, ml_confidence: float) -> dict:
    should_escalate = priority.lower() == "high" and (
        urgency_score >= 2 or (ml_confidence is not None and ml_confidence < 0.5)
    )
    reason = (
        "High priority combined with strong urgency signals or low ML confidence."
        if should_escalate
        else "No strong combined signal for escalation."
    )
    return {"should_escalate": should_escalate, "reason": reason}


TOOL_REGISTRY = {
    "get_ml_confidence": get_ml_confidence,
    "analyze_urgency_signals": analyze_urgency_signals,
    "extract_ticket_entities": extract_ticket_entities,
    "decide_escalation": decide_escalation,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_ml_confidence",
            "description": "Get the ML models' confidence scores for Issue Type and Priority predictions on this ticket.",
            "parameters": {
                "type": "object",
                "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
                "required": ["subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_urgency_signals",
            "description": "Scan the ticket text for urgency-indicating keywords and return a score.",
            "parameters": {
                "type": "object",
                "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
                "required": ["subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_ticket_entities",
            "description": "Extract reference numbers, emails, and dates mentioned in the ticket text.",
            "parameters": {
                "type": "object",
                "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
                "required": ["subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "decide_escalation",
            "description": "Decide whether this ticket should be escalated. Call AFTER you have urgency and confidence results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "priority": {"type": "string"},
                    "urgency_score": {"type": "integer"},
                    "ml_confidence": {"type": "number"},
                },
                "required": ["priority", "urgency_score", "ml_confidence"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 5. JSON VALIDATION
# ---------------------------------------------------------------------------


def parse_llama_json(raw_output):
    if not isinstance(raw_output, str):
        raise TypeError("Llama output must be a string.")

    cleaned = raw_output.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace == -1 or last_brace == -1:
        raise ValueError("No JSON object found in output.")

    parsed = json.loads(cleaned[first_brace : last_brace + 1])

    missing_fields = [f for f in REQUIRED_GENAI_FIELDS if f not in parsed]
    if missing_fields:
        raise ValueError(f"Missing fields: {missing_fields}")

    parsed_queue = str(parsed["predicted_queue"]).strip()
    if parsed_queue not in ALLOWED_QUEUES:
        raise ValueError(f"Invalid queue returned: {parsed_queue}")

    result = {field: str(parsed[field]).strip() for field in REQUIRED_GENAI_FIELDS}
    result["escalate"] = bool(parsed.get("escalate", False))
    return result


def build_correction_prompt(bad_output, error_message):
    return (
        f"Your previous response was invalid.\n\n"
        f"Previous response:\n{bad_output}\n\n"
        f"Validation error:\n{error_message}\n\n"
        f"Fix the issue and return ONLY a valid JSON object with the exact "
        f"required structure. No markdown, no extra text."
    )


# ---------------------------------------------------------------------------
# 6. AGENT ORCHESTRATOR (Groq chat.completions, OpenAI-compatible tool calling)
# ---------------------------------------------------------------------------


def build_agent_system_prompt(predicted_type, predicted_priority):
    allowed_queues_text = "\n".join(f"- {q}" for q in ALLOWED_QUEUES)
    return f"""
You are an autonomous AI agent for a customer support ticket triage system.

You have access to tools that inspect the ticket and our ML models. Use them
whenever they would help you decide better, especially before recommending
escalation. You may call multiple tools, one at a time, before answering.

ML PREDICTIONS (already computed, do not change them):
Issue Type: {predicted_type}
Priority: {predicted_priority}

ALLOWED QUEUES (choose exactly one):
{allowed_queues_text}

When you are done gathering information, respond with ONLY this JSON object
and nothing else (no markdown, no explanation):

{{
  "predicted_queue": "one exact Queue name from the allowed list",
  "summary": "a concise 1-2 sentence summary",
  "main_problem": "the main customer problem",
  "recommended_action": "the recommended next action for the selected support team",
  "suggested_response": "a short professional response to the customer",
  "escalate": true or false
}}
""".strip()


def call_groq_chat(messages, tools=None):
    kwargs = dict(
        model=LLAMA_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
    )
    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message


def run_agentic_pipeline(subject, body, log_callback=None):
    ml_result = predict_ticket_labels(subject, body)

    system_prompt = build_agent_system_prompt(
        predicted_type=ml_result["predicted_type"],
        predicted_priority=ml_result["predicted_priority"],
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Subject: {subject}\n\nBody: {body}"},
    ]

    final_raw_output = None

    for step in range(1, MAX_AGENT_STEPS + 1):
        assistant_message = call_groq_chat(messages, tools=TOOL_SCHEMAS)
        tool_calls = assistant_message.tool_calls

        if tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tc in tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                if log_callback:
                    log_callback(f"Step {step}: calling tool `{tool_name}` with {tool_args}")

                if tool_name not in TOOL_REGISTRY:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        tool_result = TOOL_REGISTRY[tool_name](**tool_args)
                    except Exception as e:
                        tool_result = {"error": str(e)}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            continue

        final_raw_output = assistant_message.content or ""
        break
    else:
        raise RuntimeError(f"Agent exceeded {MAX_AGENT_STEPS} steps without a final answer.")

    try:
        genai_result = parse_llama_json(final_raw_output)
    except (ValueError, TypeError, json.JSONDecodeError) as e:
        if log_callback:
            log_callback(f"Final answer failed validation ({e}), asking agent to fix it...")
        messages.append({"role": "user", "content": build_correction_prompt(final_raw_output, str(e))})
        fixed_message = call_groq_chat(messages)
        genai_result = parse_llama_json(fixed_message.content or "")

    return {
        "subject": subject,
        "body": body,
        "predicted_type": ml_result["predicted_type"],
        "predicted_priority": ml_result["predicted_priority"],
        **genai_result,
    }


# ---------------------------------------------------------------------------
# 7. STREAMLIT UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Support Ticket Triage", page_icon="🎫", layout="centered")

st.title("🎫 AI Support Ticket Triage")
st.caption("ML classification + an agentic GenAI layer (tool calling) running on Groq's cloud Llama.")

with st.form("ticket_form"):
    subject = st.text_input("Subject", placeholder="e.g. Incorrect invoice amount")
    body = st.text_area(
        "Body",
        placeholder="e.g. Urgent — the amount shown on my latest invoice is wrong. Please review the charges.",
        height=120,
    )
    submitted = st.form_submit_button("Process Ticket")

if submitted:
    if not subject.strip() and not body.strip():
        st.warning("Please enter a subject or body.")
    else:
        log_box = st.empty()
        logs = []

        def log_callback(msg):
            logs.append(msg)
            log_box.info("\n\n".join(logs))

        with st.spinner("Running ML models + agentic reasoning..."):
            try:
                result = run_agentic_pipeline(subject, body, log_callback=log_callback)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

        st.success("Done.")

        col1, col2 = st.columns(2)
        col1.metric("Issue Type", result["predicted_type"])
        col2.metric("Priority", result["predicted_priority"])

        st.subheader("Queue")
        st.write(result["predicted_queue"])

        if result.get("escalate"):
            st.error("⚠️ Recommended for escalation")

        st.subheader("Summary")
        st.write(result["summary"])

        st.subheader("Main Problem")
        st.write(result["main_problem"])

        st.subheader("Recommended Action")
        st.write(result["recommended_action"])

        st.subheader("Suggested Customer Response")
        st.write(result["suggested_response"])

        with st.expander("Full JSON result"):
            st.json(result)
