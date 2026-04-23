"""
main.py
Entry point for the AutoStream Conversational AI Agent.
Run: python main.py
"""

import os
import sys
from langchain_core.messages import HumanMessage

# Make sure package imports work regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

from agent.graph import build_graph, AgentState

BANNER = """
╔══════════════════════════════════════════════════════╗
║        AutoStream AI Sales Agent                     ║
║  Powered by OpenAI + LangGraph + RAG                 ║
║  Type 'quit' or 'exit' to end the session.           ║
╚══════════════════════════════════════════════════════╝
"""


def run_agent() -> None:
    """Interactive CLI loop that drives the LangGraph agent."""
    print(BANNER)

    graph = build_graph()

    # Initial state
    state: AgentState = {
        "messages": [],
        "stage": "chat",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "intent": None,
    }

    # Greet the user
    print("Agent: Hi! Welcome to AutoStream. How can I help you today?\n")

    while True:
        # Guard: if we've already finished, stop
        if state.get("stage") == "done":
            print("Agent: Thanks for chatting! Have a great day.")
            break

        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAgent: Quitting!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print("Agent: Thanks for chatting! Goodbye!")
            break

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Invoke the graph — the router entry node dispatches
        # to the correct stage-specific node automatically
        result = graph.invoke(state, config={"configurable": {"thread_id": "1"}})
        state = result

        # Print the latest AI message
        ai_messages = [m for m in state["messages"] if hasattr(m, "type") and m.type == "ai"]
        if ai_messages:
            print(f"\nAgent: {ai_messages[-1].content}\n")


if __name__ == "__main__":
    run_agent()
