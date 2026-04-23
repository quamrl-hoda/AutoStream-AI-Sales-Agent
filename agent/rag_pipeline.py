"""
rag_pipeline.py
Loads the AutoStream knowledge base from JSON and provides
a simple retrieval function for the agent to query.
"""

import json
import os
from typing import Optional

KB_PATH = os.path.join(os.path.dirname(__file__), "../knowledge_base/autostream_kb.json")


def load_knowledge_base() -> dict:
    """Load the JSON knowledge base from disk."""
    with open(KB_PATH, "r") as f:
        return json.load(f)


def retrieve_context(query: str) -> str:
    """
    Retrieves the most relevant context from the knowledge base
    based on simple keyword matching against the query.

    Returns a plain-text summary relevant to the query to be
    injected into the LLM system prompt.
    """
    kb = load_knowledge_base()
    query_lower = query.lower()

    sections = []

    # Always include company overview
    sections.append(
        f"Company: {kb['company']} — {kb['tagline']}"
    )

    # Pricing section — triggered by pricing/plan/cost/price keywords
    if any(kw in query_lower for kw in ["price", "pricing", "plan", "cost", "basic", "pro", "subscription", "pay", "cheap", "affordable", "how much"]):
        basic = kb["plans"]["basic"]
        pro = kb["plans"]["pro"]
        sections.append(
            f"\n--- PRICING ---\n"
            f"Basic Plan: ${basic['price_monthly']}/month\n"
            f"  Features: {', '.join(basic['features'])}\n\n"
            f"Pro Plan: ${pro['price_monthly']}/month\n"
            f"  Features: {', '.join(pro['features'])}"
        )

    # Refund / policy section
    if any(kw in query_lower for kw in ["refund", "cancel", "money back", "policy", "support", "help", "24/7"]):
        policies = kb["policies"]
        sections.append(
            f"\n--- POLICIES ---\n"
            f"Refund Policy: {policies['refund']}\n"
            f"Support Policy: {policies['support']}"
        )

    # Feature comparison
    if any(kw in query_lower for kw in ["4k", "720p", "caption", "resolution", "unlimited", "video", "feature", "difference", "compare"]):
        basic = kb["plans"]["basic"]
        pro = kb["plans"]["pro"]
        sections.append(
            f"\n--- FEATURES ---\n"
            f"Basic: {', '.join(basic['features'])}\n"
            f"Pro: {', '.join(pro['features'])}"
        )

    # FAQs — always append as fallback context
    faqs_text = "\n--- FAQs ---\n"
    for faq in kb["faqs"]:
        faqs_text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"
    sections.append(faqs_text)

    return "\n".join(sections)
