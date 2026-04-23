"""
intent_classifier.py
Classifies user messages into one of three intent categories:
  1. GREETING     — casual hello / small talk
  2. INQUIRY      — product / pricing questions
  3. HIGH_INTENT  — ready to sign up / strong buying signal
"""

from enum import Enum
from typing import Optional


class Intent(str, Enum):
    GREETING = "greeting"
    INQUIRY = "inquiry"
    HIGH_INTENT = "high_intent"


# Simple keyword heuristics used as a first-pass classifier.
# The LLM also classifies intent in its structured response,
# so this is a fallback / double-check layer.

_GREETING_SIGNALS = [
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "howdy", "what's up", "sup", "greetings",
]

_HIGH_INTENT_SIGNALS = [
    "sign up", "signup", "subscribe", "want to try", "want to buy",
    "let's go", "get started", "i'm in", "count me in", "ready to start",
    "i want the pro", "i want the basic", "purchase", "buy now",
    "upgrade", "onboard", "register", "create account", "join",
]

_INQUIRY_SIGNALS = [
    "price", "pricing", "cost", "plan", "feature", "how much",
    "what is", "tell me about", "explain", "does it", "can it",
    "difference", "compare", "refund", "support", "resolution",
    "4k", "720p", "caption", "video", "unlimited",
]


def classify_intent(message: str, llm_intent: Optional[str] = None) -> Intent:
    """
    Classify the intent of a user message.

    Priority:
      1. If `llm_intent` is provided (from the LLM's own classification),
         use that — it is most accurate.
      2. Fallback to keyword heuristics.

    Args:
        message:    Raw user message text.
        llm_intent: Optional intent string returned by the LLM.

    Returns:
        Intent enum value.
    """
    # Trust the LLM classification first
    if llm_intent:
        mapping = {
            "greeting": Intent.GREETING,
            "inquiry": Intent.INQUIRY,
            "high_intent": Intent.HIGH_INTENT,
        }
        intent = mapping.get(llm_intent.lower().strip())
        if intent:
            return intent

    # Fallback: keyword matching
    msg = message.lower()

    # Check HIGH_INTENT first (strongest signal)
    if any(sig in msg for sig in _HIGH_INTENT_SIGNALS):
        return Intent.HIGH_INTENT

    # Check GREETING
    tokens = set(msg.split())
    if any(sig in tokens or sig in msg for sig in _GREETING_SIGNALS):
        # Only pure greeting if message is short and has no inquiry words
        if not any(sig in msg for sig in _INQUIRY_SIGNALS) and len(msg.split()) < 10:
            return Intent.GREETING

    # Default to INQUIRY
    return Intent.INQUIRY
