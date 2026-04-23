"""
graph.py
LangGraph-based agentic workflow for the AutoStream Conversational AI Agent.

State machine stages:
  CHAT       → Normal conversation / RAG-powered Q&A
  COLLECT    → Gathering lead details (name → email → platform)
  CAPTURE    → Firing the mock_lead_capture tool
  DONE       → Conversation complete

The graph persists the full conversation history and lead-collection
progress across turns using a typed TypedDict state.
"""

import json
import re
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.rag_pipeline import retrieve_context
from agent.intent_classifier import classify_intent, Intent
from tools.lead_capture import mock_lead_capture
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# STATE SCHEMA
# ──────────────────────────────────────────────

class AgentState(TypedDict):
    # Full conversation history (auto-merged by LangGraph)
    messages: Annotated[list, add_messages]

    # Lead collection progress
    stage: Literal["chat", "collect_name", "collect_email", "collect_platform", "capture", "done"]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    # Last detected intent
    intent: Optional[str]


# ──────────────────────────────────────────────
# LLM SETUP
# ──────────────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    """Initialise the OpenAI LLM."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=512,
        temperature=0.3,
    )


llm = _build_llm()


# ──────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ──────────────────────────────────────────────

SYSTEM_BASE = """You are a helpful sales assistant for AutoStream, a SaaS platform that provides
automated video editing tools for content creators.

Your goals:
1. Greet users warmly for casual messages.
2. Answer product and pricing questions accurately using ONLY the knowledge base provided.
3. Detect when a user is ready to sign up (high intent) and begin lead qualification.
4. Collect the user's name, email, and creator platform ONE AT A TIME — never ask for all
   three at once, and never skip ahead.
5. Never call any lead-capture tool until you have all three pieces of information.

At the end of your reply, always include a JSON block like:
```json
{"intent": "<greeting|inquiry|high_intent>"}
```

Be friendly, concise, and professional.
"""


def _build_system_prompt(user_message: str, stage: str) -> str:
    context = retrieve_context(user_message)
    stage_instruction = {
        "chat": "You are in normal conversation mode. Detect intent and respond accordingly.",
        "collect_name": "You are collecting the user's FULL NAME for lead registration. Ask for it politely.",
        "collect_email": "You have the user's name. Now collect their EMAIL ADDRESS.",
        "collect_platform": "You have name and email. Now ask which CREATOR PLATFORM they use (e.g. YouTube, Instagram).",
        "capture": "You have all details. Thank the user and confirm their registration.",
        "done": "Registration is complete. Wrap up the conversation warmly.",
    }.get(stage, "")

    return f"{SYSTEM_BASE}\n\n{stage_instruction}\n\n--- KNOWLEDGE BASE ---\n{context}"


# ──────────────────────────────────────────────
# HELPER: EXTRACT INTENT FROM LLM RESPONSE
# ──────────────────────────────────────────────

def _extract_intent(response_text: str) -> Optional[str]:
    """Parse the JSON intent block from the LLM's reply."""
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("intent")
        except json.JSONDecodeError:
            pass
    return None


def _clean_response(response_text: str) -> str:
    """Remove the JSON intent block from the visible reply."""
    return re.sub(r'```json\s*\{.*?\}\s*```', '', response_text, flags=re.DOTALL).strip()


# ──────────────────────────────────────────────
# GRAPH NODES
# ──────────────────────────────────────────────

def chat_node(state: AgentState) -> AgentState:
    """
    Main conversation node.
    Handles greeting and inquiry intents; detects high-intent to
    transition to lead collection.
    """
    user_msg = state["messages"][-1].content
    system = _build_system_prompt(user_msg, "chat")

    llm_messages = [SystemMessage(content=system)] + state["messages"]
    response = llm.invoke(llm_messages)
    raw_text = response.content

    llm_intent = _extract_intent(raw_text)
    intent = classify_intent(user_msg, llm_intent)
    clean_text = _clean_response(raw_text)

    new_stage = "chat"
    if intent == Intent.HIGH_INTENT:
        new_stage = "collect_name"
        # Append a follow-up prompt to collect name
        clean_text = clean_text + "\n\nTo get you started, could I get your **full name**?"

    return {
        **state,
        "messages": [AIMessage(content=clean_text)],
        "stage": new_stage,
        "intent": intent.value,
    }


def collect_name_node(state: AgentState) -> AgentState:
    """Accept the user's name and advance to email collection."""
    user_msg = state["messages"][-1].content.strip()
    system = _build_system_prompt(user_msg, "collect_name")

    # Heuristic: if the message looks like a name (short, no @), accept it
    if len(user_msg.split()) >= 1 and "@" not in user_msg and len(user_msg) < 60:
        name = user_msg
        reply = f"Thanks, {name}! Now, what's your **email address**?"
        return {
            **state,
            "messages": [AIMessage(content=reply)],
            "stage": "collect_email",
            "lead_name": name,
        }

    # Otherwise ask again via LLM
    llm_messages = [SystemMessage(content=system)] + state["messages"]
    response = llm.invoke(llm_messages)
    clean_text = _clean_response(response.content)
    return {**state, "messages": [AIMessage(content=clean_text)]}


def collect_email_node(state: AgentState) -> AgentState:
    """Accept the user's email and advance to platform collection."""
    user_msg = state["messages"][-1].content.strip()

    email_pattern = r"[\w\.-]+@[\w\.-]+\.\w{2,}"
    match = re.search(email_pattern, user_msg)

    if match:
        email = match.group(0)
        reply = (
            f"Got it — **{email}**. Last question: which creator platform do you mainly use? "
            f"(e.g. YouTube, Instagram, TikTok, etc.)"
        )
        return {
            **state,
            "messages": [AIMessage(content=reply)],
            "stage": "collect_platform",
            "lead_email": email,
        }

    # Invalid email — ask again
    reply = "That doesn't look like a valid email address. Could you double-check and share it again?"
    return {**state, "messages": [AIMessage(content=reply)]}


def collect_platform_node(state: AgentState) -> AgentState:
    """Accept the creator platform and advance to lead capture."""
    user_msg = state["messages"][-1].content.strip()

    known_platforms = ["youtube", "instagram", "tiktok", "facebook", "twitter", "linkedin", "twitch"]
    msg_lower = user_msg.lower()

    platform = None
    for p in known_platforms:
        if p in msg_lower:
            platform = p.capitalize()
            break
    if not platform and len(user_msg) < 40:
        platform = user_msg  # Accept free-form answer

    if platform:
        return {
            **state,
            "messages": [AIMessage(content="Perfect! Let me get you registered...")],
            "stage": "capture",
            "lead_platform": platform,
        }

    reply = "Which platform do you primarily create content on? (e.g. YouTube, Instagram, TikTok)"
    return {**state, "messages": [AIMessage(content=reply)]}


def capture_node(state: AgentState) -> AgentState:
    """Fire the mock lead capture tool and conclude the conversation."""
    result = mock_lead_capture(
        name=state["lead_name"],
        email=state["lead_email"],
        platform=state["lead_platform"],
    )

    reply = (
        f"🎉 You're all set, {state['lead_name']}! We've registered your interest in AutoStream's Pro plan. "
        f"Our team will reach out to **{state['lead_email']}** shortly with next steps. "
        f"Welcome aboard, and happy creating on {state['lead_platform']}!"
    )

    return {
        **state,
        "messages": [AIMessage(content=reply)],
        "stage": "done",
    }


# ──────────────────────────────────────────────
# ROUTING LOGIC
# ──────────────────────────────────────────────

def route(state: AgentState) -> str:
    """Route to the correct node based on current stage."""
    return state.get("stage", "chat")


# ──────────────────────────────────────────────
# GRAPH ASSEMBLY
# ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("chat", chat_node)
    builder.add_node("collect_name", collect_name_node)
    builder.add_node("collect_email", collect_email_node)
    builder.add_node("collect_platform", collect_platform_node)
    builder.add_node("capture", capture_node)

    builder.set_entry_point("chat")

    # After chat node, re-route based on updated stage
    builder.add_conditional_edges(
        "chat",
        lambda s: s["stage"],
        {
            "chat": END,
            "collect_name": END,  # reply already appended in node; user types name next turn
        },
    )

    builder.add_conditional_edges(
        "collect_name",
        lambda s: s["stage"],
        {
            "collect_name": END,
            "collect_email": END,
        },
    )

    builder.add_conditional_edges(
        "collect_email",
        lambda s: s["stage"],
        {
            "collect_email": END,
            "collect_platform": END,
        },
    )

    builder.add_conditional_edges(
        "collect_platform",
        lambda s: s["stage"],
        {
            "collect_platform": END,
            "capture": "capture",
        },
    )

    builder.add_edge("capture", END)

    return builder.compile()
