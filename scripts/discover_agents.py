"""
Discover all agents with a deployment.yaml and output a GitHub Actions matrix.

Usage:
  python scripts/discover_agents.py [agent_name|all|name1,name2] [--eval-only|--traffic-only]

  agent_name     single agent, "all", or comma-separated list (e.g. "foo,bar")
  --eval-only    include only agents that have eval/eval.yaml (for eval workflows)
  --traffic-only include only agents that have traffic.py (for synthetic traffic)

Output: {"include": [{"agent": "...", "agent_image_name": "...", "agent_image_product": "...",
                      "agent_image_capability": "...", "langsmith_project": "...",
                      "eval_config": "..."}, ...]}
"""

import glob
import json
import os
import sys

import yaml

MAX_MATRIX_JOBS = 256


class DiscoveryError(Exception):
    """Raised when an agent is misconfigured (e.g. eval required but missing)."""


def collect_agents(filter_agents=None, eval_only=False, traffic_only=False):
    """Return the list of agent matrix entries under agents/*/deployment.yaml.

    filter_agents  None (means "all") or a set of agent directory names to keep.
    eval_only      when True, drop agents that have no eval/eval.yaml (and are
                   allowed to skip evals via eval_required: false).
    traffic_only   when True, drop agents that have no traffic.py (for the
                   synthetic-traffic workflow).

    Raises DiscoveryError if an agent requires evals but has no eval/eval.yaml.
    Shared by discover_agents (matrix output) and generate_deploy_all_agents.
    """
    agents = []
    for path in sorted(glob.glob("agents/*/deployment.yaml")):
        agent_name = os.path.basename(os.path.dirname(path))

        # _shared is a utility directory, never an agent.
        if agent_name == "_shared":
            continue

        if filter_agents is not None and agent_name not in filter_agents:
            continue

        # Traffic filter is a pure file check — apply it before eval validation
        # so a non-traffic agent never trips the eval-required error here.
        if traffic_only and not os.path.exists(f"agents/{agent_name}/traffic.py"):
            continue

        with open(path) as f:
            config = yaml.safe_load(f)

        image_name = config.get("image_name", agent_name.replace("_", "-"))
        meta = config.get("metadata", {})
        entry = {
            "agent": agent_name,
            "agent_image_name": image_name,
            "agent_image_product": meta["product"],
            "agent_image_capability": meta["capability"],
            "langsmith_project": config.get("langsmith_project", f"{image_name}-online"),
        }

        eval_config = f"agents/{agent_name}/eval/eval.yaml"
        if os.path.exists(eval_config):
            entry["eval_config"] = eval_config
        else:
            eval_required = config.get("eval_required", True)
            # Traffic discovery doesn't care about eval config, so a missing
            # eval.yaml must not fail it — only eval/deploy paths enforce this.
            if eval_required and not traffic_only:
                raise DiscoveryError(
                    f"{agent_name} has deployment.yaml but no eval/eval.yaml. "
                    f"Add eval/eval.yaml or set eval_required: false in deployment.yaml."
                )
            if eval_only:
                continue

        agents.append(entry)

    return agents


def main():
    args = sys.argv[1:]
    eval_only = "--eval-only" in args
    traffic_only = "--traffic-only" in args
    positional = [a for a in args if not a.startswith("--")]

    # Build the filter set from comma-separated or space-separated names.
    if not positional or "all" in positional:
        filter_agents = None  # None means "all"
    else:
        filter_agents = set()
        for p in positional:
            filter_agents.update(x.strip() for x in p.split(",") if x.strip())

    try:
        agents = collect_agents(
            filter_agents=filter_agents, eval_only=eval_only, traffic_only=traffic_only
        )
    except DiscoveryError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if not agents:
        filter_desc = ",".join(sorted(filter_agents)) if filter_agents else "all"
        print(
            f"No agents found (filter: {filter_desc}, eval_only: {eval_only}, "
            f"traffic_only: {traffic_only})",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(agents) > MAX_MATRIX_JOBS:
        print(
            f"ERROR: matrix has {len(agents)} jobs, exceeds GitHub's {MAX_MATRIX_JOBS} cap. "
            f"Shard the matrix or split the run.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(json.dumps({"include": agents}))


if __name__ == "__main__":
    main()
