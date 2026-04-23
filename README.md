# AutoStream AI Sales Agent 🎬

> A Conversational AI Agent built with **LangGraph +  Openai + RAG**  
> Assignment Project for **ServiceHive / Inflx** — Social-to-Lead Agentic Workflow

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [How to Run Locally](#how-to-run-locally)
4. [Architecture Explanation](#architecture-explanation)
5. [Conversation Flow](#conversation-flow)
6. [WhatsApp Deployment via Webhooks](#whatsapp-deployment-via-webhooks)
7. [Running Tests](#running-tests)
8. [Tech Stack](#tech-stack)

---

## Overview

This agent acts as an AI-powered sales assistant for **AutoStream**, a fictional SaaS platform offering automated video editing tools. The agent can:

- Classify user intent: **Greeting / Product Inquiry / High-Intent Lead**
- Answer questions using a **RAG-powered local knowledge base**
- Detect buying intent and begin **lead qualification**
- Collect Name → Email → Platform **one at a time**
- Call a **mock lead-capture API** only after all three fields are collected
- Maintain **conversation memory** across 5–6 turns using LangGraph state

---

## Project Structure

```
autostream_agent/
│
├── main.py                        # Entry point — interactive CLI loop
│
├── agent/
│   ├── graph.py                   # LangGraph state machine (nodes + edges)
│   ├── rag_pipeline.py            # Knowledge base loader + retrieval
│   └── intent_classifier.py       # Intent classification (keyword + LLM)
│
├── tools/
│   └── lead_capture.py            # mock_lead_capture() function
│
├── knowledge_base/
│   └── autostream_kb.json         # Pricing, features, policies (RAG source)
│
├── tests/
│   └── test_agent.py              # Unit tests (pytest)
│
├── requirements.txt
├── .env.example
└── README.md
```

## How to Run Locally

### Prerequisites

- Python 3.9 or later
- An **Anthropic API key** (free tier works) — get one at https://console.anthropic.com/

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/AutoStream-AI-Sales-Agent.git
cd AutoStream-AI-Sales-Agent

# 2. Create and activate a virtual environment (recommended)
uv install
uv init
uv venv 
or 
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# Open .env and replace "your_api_key_here" with your actual key
# OPENAI_API_KEY=...

# 5. Run the agent
uv run main.py
```

You should see:

```
╔══════════════════════════════════════════════════════╗
║        AutoStream AI Sales Agent  🎬                 ║
║  Powered by OpenAI + LangGraph + RAG           ║
╚══════════════════════════════════════════════════════╝

Agent: Hi! Welcome to AutoStream. How can I help you today?

You: _
```

Type naturally and the agent will respond. Type `quit` or `exit` to end.

---

## Architecture Explanation

### Why LangGraph?

LangGraph was chosen over AutoGen because this workflow is **sequential and stage-driven** — the agent must move through defined phases (chat → collect name → collect email → collect platform → capture) without allowing the LLM to skip steps or fire the lead-capture tool prematurely.

LangGraph's **typed state machine** enforces these transitions explicitly:

- Each stage is a named **node** with its own prompt and logic
- **Conditional edges** route between nodes based on the current `stage` field in state
- The `AgentState` TypedDict is the single source of truth — it holds the full message history, lead fields, current stage, and last detected intent

AutoGen's agent-loop model would require more guardrails to prevent premature tool invocation, making LangGraph a cleaner fit here.

### State Management

```
AgentState (TypedDict)
  ├── messages        → Full conversation history (appended each turn)
  ├── stage           → Current workflow stage (chat / collect_name / ... / done)
  ├── lead_name       → Collected name (None until provided)
  ├── lead_email      → Collected email (None until provided)
  ├── lead_platform   → Collected platform (None until provided)
  └── intent          → Last detected intent string
```

State persists across all turns in the CLI session. LangGraph merges new messages into the history automatically via the `add_messages` annotation. The `stage` field acts as a deterministic router, so the LLM can never jump to lead capture unless all three fields are populated.

### RAG Pipeline

The knowledge base lives in `knowledge_base/autostream_kb.json`. On each turn, `retrieve_context(user_message)` performs lightweight keyword-based retrieval to select relevant sections (pricing, policies, features, FAQs) and injects them into the system prompt. This approach keeps latency low and avoids the overhead of a vector database for this small knowledge base.

### Intent Detection

Intent classification happens in two layers:
1. The **LLM** includes a small JSON block in every reply classifying intent as `greeting`, `inquiry`, or `high_intent`.
2. A **keyword heuristic** in `intent_classifier.py` acts as a fallback or double-check.

The LLM classification takes priority; heuristics fire only when the LLM's JSON block is missing or unparsable.

---

## Conversation Flow

```
User: "Hi, tell me about your pricing."
  → Intent: INQUIRY
  → RAG retrieves pricing section
  → Agent explains Basic ($29/mo) and Pro ($79/mo) plans

User: "That sounds great, I want to try the Pro plan for my YouTube channel."
  → Intent: HIGH_INTENT
  → Stage transitions to: collect_name

Agent: "To get you started, could I get your full name?"

User: "Priya Sharma"
  → Name saved → Stage: collect_email

Agent: "Thanks, Priya! What's your email address?"

User: "priya@example.com"
  → Email validated → Stage: collect_platform

Agent: "Got it. Which creator platform do you mainly use?"

User: "YouTube"
  → Platform saved → Stage: capture
  → mock_lead_capture("Priya Sharma", "priya@example.com", "YouTube") called
  → Stage: done

Agent: "🎉 You're all set, Priya! Our team will reach out to priya@example.com shortly."
```

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, follow these steps:

### 1. Set Up a WhatsApp Business Account
- Register for the **WhatsApp Business API** via Meta for Developers (https://developers.facebook.com/docs/whatsapp)
- Create an App, add the WhatsApp product, and obtain your **Phone Number ID** and **Access Token**

### 2. Build a Webhook Server

Create a simple HTTP server (Flask or FastAPI) that:

```python
from flask import Flask, request, jsonify
from main import build_graph_session   # expose a session factory

app = Flask(__name__)
sessions = {}  # In-memory store; use Redis in production

@app.route("/webhook", methods=["GET"])
def verify():
    # WhatsApp sends a GET to verify the webhook
    if request.args.get("hub.verify_token") == "YOUR_VERIFY_TOKEN":
        return request.args.get("hub.challenge")
    return "Unauthorized", 403

@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.json
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            message = change["value"]["messages"][0]
            sender = message["from"]          # WhatsApp phone number
            text   = message["text"]["body"]  # User's message

            # Retrieve or create session state
            state = sessions.get(sender, initial_state())
            state, reply = process_turn(state, text)
            sessions[sender] = state

            # Send reply back via WhatsApp Cloud API
            send_whatsapp_message(sender, reply)

    return jsonify({"status": "ok"})
```

### 3. Send Replies via WhatsApp Cloud API

```python
import requests

def send_whatsapp_message(to: str, body: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body}
    }
    requests.post(url, json=payload, headers=headers)
```

### 4. Register the Webhook

In the Meta Developer Console, set your webhook URL to `https://yourdomain.com/webhook` and configure the verify token.

### 5. Production Considerations

| Concern | Recommendation |
|---|---|
| Session state | Replace in-memory dict with **Redis** (keyed by phone number) |
| Scaling | Deploy on **AWS Lambda** or **Cloud Run** (stateless, event-driven) |
| Security | Validate `X-Hub-Signature-256` header on every incoming POST |
| Media | Use WhatsApp's media API to send images/videos if the agent needs them |
| Rate limits | WhatsApp allows ~1,000 free conversations/month on the basic tier |

---

## Running Tests

```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
python -m pytest tests/test_agent.py -v
```

Expected output:
```
tests/test_agent.py::TestIntentClassifier::test_greeting         PASSED
tests/test_agent.py::TestIntentClassifier::test_pricing_inquiry  PASSED
tests/test_agent.py::TestIntentClassifier::test_high_intent_signup PASSED
...
tests/test_agent.py::TestLeadCapture::test_successful_capture    PASSED
tests/test_agent.py::TestLeadCapture::test_invalid_email_raises  PASSED
...
8 passed in 0.42s
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Agent Framework | LangGraph 0.1+ |
| LLM | Claude Haiku (claude-haiku-4-5) via Anthropic API |
| LLM Abstraction | LangChain + langchain-anthropic |
| Knowledge Base | Local JSON file (RAG via keyword retrieval) |
| Lead Tool | Custom `mock_lead_capture()` function |
| Testing | pytest |
| Deployment (optional) | Flask/FastAPI + WhatsApp Cloud API |

---

*Built for the ServiceHive / Inflx ML Intern Assignment — AutoStream Social-to-Lead Agentic Workflow*
