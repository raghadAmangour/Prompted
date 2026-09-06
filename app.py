"""
AI Support Ticket Triage — Streamlit App
ML (Issue Type + Priority) + Agentic GenAI layer (Groq-hosted Llama, tool calling)

Deploy notes:
- Put issue_type_model.joblib and priority_model.joblib inside the models/ folder
  next to this file (NOT an external path like ../Downloads/...).
- Set GROQ_API_KEY in Streamlit Cloud's "Secrets" panel (or a local .streamlit/secrets.toml):
    GROQ_API_KEY = "gsk_..."
"""

import html
import json
import re
import textwrap

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

st.set_page_config(page_title="Ticket Triage Console", page_icon="🎫", layout="centered")

# Developer/debug mode: OFF by default for every normal visit.
# To inspect the raw JSON while testing, open the app with ?debug=1 added
# to the URL, e.g. http://localhost:8501/?debug=1
DEBUG_MODE = st.query_params.get("debug") == "1"

# --- Design tokens -----------------------------------------------------
# A dark "ops console" palette: the product reads support tickets the way
# an engineer reads a monitoring dashboard, so the chrome borrows from that
# world (deep ink background, hairline borders, monospace system labels)
# instead of a generic SaaS card kit. Priority is the one place color
# carries meaning — every other surface stays quiet.
st.markdown(
    textwrap.dedent(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --bg: #0B1220;
        --panel: #121B2E;
        --panel-2: #16213A;
        --border: #223049;
        --text: #EAF0F7;
        --muted: #8FA0B8;
        --accent: #4FD1C5;
        --accent-text: #06231F;
        --high: #F2545B;
        --medium: #F2A93B;
        --low: #4FD1C5;
    }

    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    .block-container { max-width: 760px; padding-top: 2.5rem; }

    /* Header / masthead */
    .console-mast { display: flex; align-items: baseline; gap: .6rem; margin-bottom: .15rem; }
    .console-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent);
                   display: inline-block; box-shadow: 0 0 6px var(--accent); }
    .console-title { font-family: 'IBM Plex Mono', monospace; font-size: 1.15rem;
                      font-weight: 600; letter-spacing: .01em; color: var(--text); }
    .console-sub { color: var(--muted); font-size: .88rem; margin-bottom: 1.6rem; }
    .console-status { display: flex; gap: 1.4rem; flex-wrap: wrap; margin-bottom: 1.8rem;
                        font-family: 'IBM Plex Mono', monospace; font-size: .74rem; color: var(--muted); }
    .console-status b { color: var(--text); font-weight: 500; }

    /* Form panel */
    [data-testid="stForm"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1.4rem 1.4rem .9rem 1.4rem;
    }
    [data-testid="stWidgetLabel"] p {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--muted);
        font-size: .78rem;
    }
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        background: var(--panel-2) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    [data-testid="stFormSubmitButton"] button {
        background: var(--accent);
        color: var(--accent-text);
        border: none;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        border-radius: 6px;
        padding: .5rem 1.2rem;
    }
    [data-testid="stFormSubmitButton"] button:hover { background: #3fb8ad; color: var(--accent-text); }

    /* Ticket result card */
    .ticket-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 4px solid var(--muted);
        border-radius: 8px;
        padding: 1.2rem 1.4rem;
        margin-top: 1.6rem;
    }
    .ticket-card-header { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; }
    .ticket-id { font-family: 'IBM Plex Mono', monospace; color: var(--muted); font-size: .82rem; }
    .badge {
        font-family: 'IBM Plex Mono', monospace; font-size: .68rem; font-weight: 600;
        padding: .2rem .55rem; border-radius: 20px; letter-spacing: .02em;
    }
    .badge-escalate { background: rgba(242,84,91,.15); color: var(--high); border: 1px solid rgba(242,84,91,.4); }
    .ticket-meta { display: flex; gap: 1.6rem; margin-top: .7rem; font-size: .86rem; color: var(--text); }
    .meta-label { font-family: 'IBM Plex Mono', monospace; color: var(--muted); font-size: .72rem;
                   display: block; margin-bottom: .1rem; }
    .ticket-divider { border: none; border-top: 1px solid var(--border); margin: .9rem 0; }
    .ticket-field { margin-bottom: .9rem; }
    .ticket-field:last-child { margin-bottom: 0; }
    .field-label { font-family: 'IBM Plex Mono', monospace; color: var(--muted); font-size: .72rem;
                    margin-bottom: .2rem; }
    .field-value { font-size: .92rem; line-height: 1.5; color: var(--text); }

    .response-caption { font-family: 'IBM Plex Mono', monospace; color: var(--muted);
                          font-size: .78rem; margin: 1.1rem 0 .4rem 0; }
    [data-testid="stCodeBlock"] pre { background: var(--panel-2) !important; border: 1px solid var(--border) !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] * { color: var(--text); }
    .sidebar-ticket { display: flex; align-items: center; gap: .5rem; padding: .35rem 0;
                        font-size: .82rem; border-bottom: 1px solid var(--border); }
    .sidebar-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .sidebar-ticket-id { font-family: 'IBM Plex Mono', monospace; color: var(--muted); font-size: .74rem; }
    </style>
    """
    ),
    unsafe_allow_html=True,
)

PRIORITY_COLOR = {"high": "var(--high)", "medium": "var(--medium)", "low": "var(--low)"}
PRIORITY_COLOR_HEX = {"high": "#F2545B", "medium": "#F2A93B", "low": "#4FD1C5"}

if "ticket_counter" not in st.session_state:
    st.session_state.ticket_counter = 8214
if "ticket_history" not in st.session_state:
    st.session_state.ticket_history = []

# --- Sidebar: how it works + session history ---------------------------
with st.sidebar:
    st.markdown("**How this engine works**")
    st.caption(
        "1. Two ML models classify Issue Type and Priority.\n\n"
        "2. A GenAI agent (Groq-hosted Llama) can call tools — urgency scan, "
        "confidence check, entity extraction, escalation logic — before answering.\n\n"
        "3. It returns the queue, summary, and a ready customer reply."
    )
    st.divider()
    st.markdown("**Recent tickets**")
    if not st.session_state.ticket_history:
        st.caption("Nothing processed yet this session.")
    else:
        for t in reversed(st.session_state.ticket_history[-8:]):
            color = PRIORITY_COLOR_HEX.get(t["priority"].lower(), "#8FA0B8")
            st.markdown(
                textwrap.dedent(
                    f"""<div class="sidebar-ticket">
                        <span class="sidebar-dot" style="background:{color}"></span>
                        <span class="sidebar-ticket-id">{html.escape(t['id'])}</span>
                        <span>{html.escape(t['subject'][:28])}</span>
                    </div>"""
                ),
                unsafe_allow_html=True,
            )

# --- Masthead ------------------------------------------------------------
st.markdown(
    textwrap.dedent(
        """
    <div class="console-mast">
        <span class="console-dot"></span>
        <span class="console-title">Ticket Triage Console</span>
    </div>
    <div class="console-sub">ML classification + an agentic GenAI layer, running on Groq's cloud Llama.</div>
    """
    ),
    unsafe_allow_html=True,
)
st.markdown(
    textwrap.dedent(
        f"""
    <div class="console-status">
        <span>ML models &nbsp;<b>ready</b></span>
        <span>Reasoning engine &nbsp;<b>{html.escape(LLAMA_MODEL)}</b></span>
        <span>Tickets this session &nbsp;<b>{len(st.session_state.ticket_history)}</b></span>
    </div>
    """
    ),
    unsafe_allow_html=True,
)

# --- Compose form ----------------------------------------------------------
with st.form("ticket_form"):
    subject = st.text_input("Subject", placeholder="e.g. Incorrect invoice amount")
    body = st.text_area(
        "Body",
        placeholder="e.g. Urgent — the amount shown on my latest invoice is wrong. Please review the charges.",
        height=120,
    )
    submitted = st.form_submit_button("Analyze ticket")

if submitted:
    if not subject.strip() and not body.strip():
        st.warning("Enter a subject or body first.")
    else:
        # Internal agent steps (tool calls, retries, etc.) are intentionally
        # not shown here — they're only useful for debugging, so we collect
        # them silently. Print `debug_logs` yourself if you need to inspect them.
        debug_logs = []

        def log_callback(msg):
            debug_logs.append(msg)

        with st.spinner("Analyzing ticket..."):
            try:
                result = run_agentic_pipeline(subject, body, log_callback=log_callback)
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                st.stop()

        st.session_state.ticket_counter += 1
        ticket_id = f"TCK-{st.session_state.ticket_counter}"
        st.session_state.ticket_history.append(
            {"id": ticket_id, "subject": subject or "(no subject)", "priority": result["predicted_priority"]}
        )

        priority_key = result["predicted_priority"].strip().lower()
        priority_color = PRIORITY_COLOR.get(priority_key, "var(--muted)")
        escalate_badge = (
            '<span class="badge badge-escalate">Escalation recommended</span>'
            if result.get("escalate")
            else ""
        )

        def esc(x):
            return html.escape(str(x))

        st.markdown(
            textwrap.dedent(
                f"""
            <div class="ticket-card" style="border-left-color:{priority_color}">
                <div class="ticket-card-header">
                    <span class="ticket-id">{ticket_id}</span>
                    <span class="badge" style="background:{priority_color}22;color:{priority_color};
                          border:1px solid {priority_color}55;">{esc(result['predicted_priority']).upper()} PRIORITY</span>
                    {escalate_badge}
                </div>
                <div class="ticket-meta">
                    <span><span class="meta-label">Type</span>{esc(result['predicted_type'])}</span>
                    <span><span class="meta-label">Queue</span>{esc(result['predicted_queue'])}</span>
                </div>
                <hr class="ticket-divider"/>
                <div class="ticket-field">
                    <div class="field-label">Summary</div>
                    <div class="field-value">{esc(result['summary'])}</div>
                </div>
                <div class="ticket-field">
                    <div class="field-label">Main problem</div>
                    <div class="field-value">{esc(result['main_problem'])}</div>
                </div>
                <div class="ticket-field">
                    <div class="field-label">Recommended action</div>
                    <div class="field-value">{esc(result['recommended_action'])}</div>
                </div>
            </div>
            """
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="response-caption">Suggested customer response</div>', unsafe_allow_html=True)
        st.code(result["suggested_response"], language=None)

        if DEBUG_MODE:
            with st.expander("Full JSON result"):
                st.json(result)
