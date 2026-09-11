# AI Support Ticket Triage System

An end-to-end support ticket triage system that combines **machine learning classification** (Issue Type + Priority) with a **Generative AI / Agentic layer** (Groq-hosted LLM with tool calling) for automated queue routing, response drafting, and escalation decisions — wrapped in a Streamlit console for support agents.

## Project Description

The system processes an incoming ticket (subject + body) through two stages:

1. **Machine Learning classification** — two independently trained scikit-learn models predict the ticket's **Issue Type** (Incident / Request / Problem / Change) and **Priority** (High / Medium / Low).
2. **Agentic AI layer** — a Groq-hosted generative model receives the ML predictions and, using four callable tools (`get_ml_confidence`, `analyze_urgency_signals`, `extract_ticket_entities`, `decide_escalation`), produces:
   - the recommended support **queue**,
   - a ticket **summary** and **main problem** statement,
   - a **recommended action**,
   - a **suggested customer response**,
   - and an **escalation flag**.

Two deterministic safety layers sit around the generative model and are never left to its judgment alone:
- A **priority safety override**: urgency keywords in the ticket text can only *raise* the ML-predicted priority, never lower it.
- A **server-side escalation decision**: the final `escalate` value is computed by rule-based logic, not accepted directly from the LLM's output.

Three additional deterministic safeguards were designed and evaluated for the generative output: a **hallucination grounding check**, an **unsafe-advice pattern filter**, and a **human-approval gate** (`requires_human_review()`).

## Dataset

- Source: [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) (Hugging Face, CC-BY-NC-4.0).
- English-language subset, cleaned of missing subjects/bodies, deduplicated, and PII-masked (emails, phone numbers, IP addresses).
- Final training set: **17,217 unique tickets** across 4 issue types, 3 priority levels, and 10 real support queue categories.

## Repository Structure

```
├── app.py               # Streamlit "Ticket Triage Console" (ML + Agentic + UI)
├── data/
│   └── download_data.py # Fetches and prepares the dataset from Hugging Face
├── modeling/            # ML training script + result summaries
├── models/              # Trained .joblib models
├── evaluation/          # Ground-truth construction, agent evaluation runner, safeguards
├── docs/                # Final report
├── requirements.txt
└── .gitignore
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download and prepare the dataset
```bash
python data/download_data.py
```
This fetches the dataset from Hugging Face and prepares it for training. Make sure the output file matches the path expected by `modeling/final_ml_pipeline.py`.

### 3. Train the ML models
```bash
cd modeling
python final_ml_pipeline.py
```
This trains and tunes 5 candidate models (Logistic Regression, Linear SVM, Multinomial Naive Bayes, SGD, Complement Naive Bayes) against a `DummyClassifier` baseline via `RandomizedSearchCV` (15 iterations, 5-fold Stratified CV), and saves the winning models to `models/issue_type_model.joblib` and `models/priority_model.joblib`.

### 4. Configure the Groq API key
Create a `secrets.toml` file (or `.streamlit/secrets.toml`) **in the project root** — **never commit this file**:
```toml
GROQ_API_KEY = "gsk_..."
```

### 5. Launch the app
Run this from the **project root** (not from inside any subfolder), since model and secrets paths are resolved relative to the working directory:
```bash
streamlit run app.py
```
Paste a ticket's subject and body, click **Analyze Ticket**, and review the classification, routing, and suggested response.

### 6. (Optional) Reproduce the evaluation
```bash
cd evaluation
python build_ground_truth.py        # builds the clean held-out ground-truth sample
python run_agent_evaluation.py      # runs the full agentic pipeline against it
```
`safeguards.py` contains the three deterministic checks (hallucination grounding, unsafe-advice filter, human-approval gate) applied to the agent's outputs; see `routing_ground_truth_50_RESULTS_with_safeguards.csv` for the applied results.

## Summarized Results

### Machine Learning

| Task | Winning Model | Metric | Result |
|---|---|---|---|
| Issue Type | Logistic Regression | Accuracy / Macro-F1 | 86.1% / 0.865 |
| Priority | Logistic Regression | Cost-sensitive penalty per ticket | 5.48 (46% reduction vs. 10.09 baseline) |
| Priority | — | High-priority tickets misclassified as Low | 107 / 1,317 (8.1%) |

Priority classification used a custom cost matrix that penalizes a genuinely High-priority ticket being misclassified as Low most heavily, since that is the costliest real-world error.

### Agentic AI Layer

Evaluated on a held-out sample (queue labels were never used during ML training, so this is leakage-free) using a 6-criterion human rubric (Factual Grounding, No Premature Completion, Queue/Type Fit, Professionalism, Actionability, Safety) and 3 deterministic safeguards.

| Metric | Target | Actual | Status |
|---|---|---|---|
| Routing Accuracy (all evaluated tickets) | ≥ 85% | 20% | Not Met* |
| Routing Accuracy (supported queue classes only) | ≥ 85% | 62.5% | Not Met |
| Response Quality (human rubric average) | ≥ 4.0 / 5 | 3.86 / 5 | Not Met |
| Hallucination Rate (human-reviewed) | ≤ 5% | 36% | Not Met |
| Unsafe Advice Rate | ≤ 1% | 2% | Not Met |
| Automated Human-Review Coverage | 100% | 8% (52% actually needed) | Not Met |
| Average Latency | ≤ 5 s | 23.8 s | Not Met** |

\* The raw routing accuracy is driven down by a mismatch between the queue labels present in the dataset (10 categories) and the queue labels currently supported by the implementation (5 categories) — not by weak agent reasoning.

\** Latency was measured on Llama 3.3 70B (Versatile) as a substitute for the intended production model (`openai/gpt-oss-20b`) due to a Groq daily-quota limitation encountered during evaluation, and should not be read as a production estimate.

**Key finding:** the automated `requires_human_review()` rule flagged only 8% of tickets for review, while human evaluation determined 52% actually needed it — and 11 of the tickets it missed contained genuine hallucinations. This shows that automated grounding checks alone are not sufficient, and human review remains a necessary safety layer for this system.

## Known Limitations & Next Steps

- Expand `ALLOWED_QUEUES` from 5 to 10 categories to match the real queue distribution in the data.
- Refine the system prompt to reduce "premature completion" language (claiming an action is done/guaranteed when it isn't) — the main driver of the hallucination rate.
- Strengthen `requires_human_review()` to close the gap between automated and human-identified review needs.
- Integrate all three safeguards directly into the live pipeline, with a mandatory "Approve & Send" gate.
- Re-evaluate latency on the intended production model once sufficient API quota is secured.
- Validate on real-world (non-synthetic) support tickets.

Full methodology, evaluation details, and per-model confusion matrices are documented in [`docs/final_report`](docs/).
