"""Agent entry points for MAP evaluation.

Two entry points share the same graph:

- ``target(inputs)``  — LangSmith evaluate() style: receives a dict, returns a dict
  with ``messages`` and ``answer``. Used by run_trajectory_evals.py.

- ``ask(question, **context)`` — MAP offline eval style: receives a plain string,
  returns ``{"answer", "tool_calls", "metadata"}``. Used by run_offline_evals.py.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from langchain_core.messages import HumanMessage

from graph import GRAPH


def target(inputs: dict) -> dict:
    result = GRAPH.invoke({"messages": [HumanMessage(content=inputs["question"])]})
    messages = result["messages"]
    return {"messages": messages, "answer": messages[-1].content}


def ask(question: str, **context) -> dict:  # noqa: ARG001 — context kept for MAP eval API compatibility
    result = GRAPH.invoke({"messages": [HumanMessage(content=question)]})
    messages = result["messages"]
    answer = messages[-1].content if messages else ""

    calls_by_id: dict[str, dict] = {}
    for message in messages:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                call_id = tc.get("id") or ""
                calls_by_id[call_id] = {
                    "tool_name": tc.get("name"),
                    "args": tc.get("args", {}),
                    "id": call_id,
                    "result": None,
                }
        if getattr(message, "type", None) == "tool":
            call_id = getattr(message, "tool_call_id", "") or ""
            if call_id in calls_by_id:
                calls_by_id[call_id]["result"] = getattr(message, "content", None)

    return {
        "answer": answer,
        "tool_calls": list(calls_by_id.values()),
        "metadata": {
            "agent_name": "platform-test-agent",
            "agent_version": "v1",
        },
    }
