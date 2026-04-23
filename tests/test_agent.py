"""
test_agent.py
Basic automated tests for individual agent components.
Run with: python -m pytest tests/test_agent.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from agent.intent_classifier import classify_intent, Intent
from agent.rag_pipeline import retrieve_context
from tools.lead_capture import mock_lead_capture


# ──────────────────────────────────────────────
# INTENT CLASSIFIER TESTS
# ──────────────────────────────────────────────

class TestIntentClassifier:
    def test_greeting(self):
        assert classify_intent("Hi there!") == Intent.GREETING

    def test_greeting_hello(self):
        assert classify_intent("Hello") == Intent.GREETING

    def test_pricing_inquiry(self):
        result = classify_intent("What are your pricing plans?")
        assert result == Intent.INQUIRY

    def test_feature_inquiry(self):
        result = classify_intent("Does the Pro plan include 4K resolution?")
        assert result == Intent.INQUIRY

    def test_high_intent_signup(self):
        result = classify_intent("I want to sign up for the Pro plan!")
        assert result == Intent.HIGH_INTENT

    def test_high_intent_try(self):
        result = classify_intent("I want to try the Pro plan for my YouTube channel.")
        assert result == Intent.HIGH_INTENT

    def test_llm_override(self):
        # LLM says high_intent even if keywords suggest inquiry
        result = classify_intent("Tell me more about pricing", llm_intent="high_intent")
        assert result == Intent.HIGH_INTENT


# ──────────────────────────────────────────────
# RAG PIPELINE TESTS
# ──────────────────────────────────────────────

class TestRAGPipeline:
    def test_retrieves_pricing_for_price_query(self):
        context = retrieve_context("What is the price of the Pro plan?")
        assert "$79" in context
        assert "$29" in context

    def test_retrieves_refund_policy(self):
        context = retrieve_context("What is your refund policy?")
        assert "7 days" in context

    def test_retrieves_features(self):
        context = retrieve_context("Does Pro include 4K and AI captions?")
        assert "4K" in context
        assert "AI captions" in context

    def test_always_includes_company_name(self):
        context = retrieve_context("hello")
        assert "AutoStream" in context


# ──────────────────────────────────────────────
# LEAD CAPTURE TOOL TESTS
# ──────────────────────────────────────────────

class TestLeadCapture:
    def test_successful_capture(self, capsys):
        result = mock_lead_capture("Jane Doe", "jane@example.com", "YouTube")
        assert result["status"] == "success"
        assert "jane@example.com" in result["message"]

    def test_returns_lead_id(self):
        result = mock_lead_capture("John Smith", "john@test.com", "Instagram")
        assert "lead_id" in result
        assert result["lead_id"].startswith("LEAD-")

    def test_invalid_email_raises(self):
        with pytest.raises(ValueError, match="Invalid email"):
            mock_lead_capture("Jane", "not-an-email", "TikTok")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            mock_lead_capture("", "jane@example.com", "YouTube")

    def test_empty_platform_raises(self):
        with pytest.raises(ValueError, match="platform cannot be empty"):
            mock_lead_capture("Jane", "jane@example.com", "")
