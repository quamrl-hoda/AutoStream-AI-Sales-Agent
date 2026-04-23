"""
test_flow.py
Automated test that simulates the exact conversation flow that was
hallucinating: sign-up -> name -> email -> platform ("striver creator platform")
Verifies the router dispatches correctly at each stage.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage
from agent.graph import build_graph, AgentState


def test_full_flow():
    graph = build_graph()

    state: AgentState = {
        "messages": [],
        "stage": "chat",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "intent": None,
    }

    # Step 1: User shows buying intent
    print("=" * 60)
    print("STEP 1: User says 'I want to sign up for the Pro plan'")
    print("=" * 60)
    state["messages"] = state["messages"] + [HumanMessage(content="I want to sign up for the Pro plan")]
    state = graph.invoke(state, config={"configurable": {"thread_id": "test1"}})

    ai_msgs = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"]
    print(f"Stage: {state['stage']}")
    print(f"Agent: {ai_msgs[-1].content}\n")
    assert state["stage"] == "collect_name", f"Expected collect_name, got {state['stage']}"

    # Step 2: User provides name
    print("=" * 60)
    print("STEP 2: User says 'Priya'")
    print("=" * 60)
    state["messages"] = state["messages"] + [HumanMessage(content="Priya")]
    state = graph.invoke(state, config={"configurable": {"thread_id": "test1"}})

    ai_msgs = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"]
    print(f"Stage: {state['stage']}")
    print(f"Lead name: {state['lead_name']}")
    print(f"Agent: {ai_msgs[-1].content}\n")
    assert state["stage"] == "collect_email", f"Expected collect_email, got {state['stage']}"
    assert state["lead_name"] == "Priya", f"Expected Priya, got {state['lead_name']}"

    # Step 3: User provides email 
    print("=" * 60)
    print("STEP 3: User says 'priya@example.com'")
    print("=" * 60)
    state["messages"] = state["messages"] + [HumanMessage(content="priya@example.com")]
    state = graph.invoke(state, config={"configurable": {"thread_id": "test1"}})

    ai_msgs = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"]
    print(f"Stage: {state['stage']}")
    print(f"Lead email: {state['lead_email']}")
    print(f"Agent: {ai_msgs[-1].content}\n")
    assert state["stage"] == "collect_platform", f"Expected collect_platform, got {state['stage']}"
    assert state["lead_email"] == "priya@example.com", f"Expected priya@example.com, got {state['lead_email']}"

    # Step 4: User provides platform ("striver creator platform") 
    print("=" * 60)
    print("STEP 4: User says 'striver creator platform'")
    print("=" * 60)
    state["messages"] = state["messages"] + [HumanMessage(content="striver creator platform")]
    state = graph.invoke(state, config={"configurable": {"thread_id": "test1"}})

    ai_msgs = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"]
    print(f"Stage: {state['stage']}")
    print(f"Lead platform: {state['lead_platform']}")
    print(f"Agent: {ai_msgs[-1].content}\n")
    assert state["stage"] == "done", f"Expected done, got {state['stage']}"
    assert "striver" in state["lead_platform"].lower(), f"Expected Striver in platform, got {state['lead_platform']}"

    print("=" * 60)
    print("ALL ASSERTIONS PASSED - No hallucination!")
    print("=" * 60)


if __name__ == "__main__":
    test_full_flow()
