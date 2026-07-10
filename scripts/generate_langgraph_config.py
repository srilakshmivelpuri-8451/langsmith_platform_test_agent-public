"""
Generate a langgraph.json for a specific agent.

Usage: python scripts/generate_langgraph_config.py <agent_name>

Reads from agents/{agent_name}/deployment.yaml:
  graph_name  - LangSmith graph name (default: agent directory name)
  graph_path  - path to graph entrypoint relative to repo root, e.g.
                "agents/my_agent/agent.py:GRAPH"
                (default: "agents/{agent_name}/graph.py:graph")
"""

import json
import os
import sys

import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_langgraph_config.py <agent_name>", file=sys.stderr)
        sys.exit(1)

    agent_name = sys.argv[1]

    with open(f"agents/{agent_name}/deployment.yaml") as f:
        deployment_config = yaml.safe_load(f)

    graph_name = deployment_config.get("graph_name", agent_name)
    graph_path = deployment_config.get("graph_path", f"agents/{agent_name}/graph.py:graph")
    config = {
        "dependencies": ["."],
        "graphs": {
            graph_name: f"./{graph_path}",
        },
        "env": ".env",
    }

    # If the agent ships an app.py, mount it as a custom HTTP app inside the
    # LangGraph Platform container. This enables the built-in chat frontend.
    app_path = f"agents/{agent_name}/app.py"
    if os.path.exists(app_path):
        config["http"] = {"app": f"./{app_path}:app"}

    with open("langgraph.json", "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    has_app = "http" in config
    print(
        f"Generated langgraph.json for agent: {agent_name} "
        f"(graph_name: {graph_name}, graph_path: {graph_path}"
        + (f", http app: {app_path}" if has_app else "")
        + ")"
    )


if __name__ == "__main__":
    main()
