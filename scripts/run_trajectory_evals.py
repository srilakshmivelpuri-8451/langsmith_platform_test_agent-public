"""
Trajectory evaluation using the effolang-agent-evals SDK.

This script wraps the agent's ask() function for use with langsmith.evaluate()
and the effolang-agent-evals deterministic + LLM-judge trajectory evaluators.

The golden dataset's `expected_tools` field is promoted to
`expected_trajectory` on-the-fly so the reference-driven evaluators
pick it up without requiring a schema change to the dataset file.

Usage:
    PYTHONPATH=. python scripts/run_trajectory_evals.py \
        --config agents/demo_grocery_agent/evals/agent_eval_config.yaml

Exit code 0 = all deterministic evaluators pass at or above threshold.
Exit code 1 = one or more evaluators failed or threshold was not met.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Config loading (mirrored from run_offline_evals.py to stay dependency-free)
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_target(target: str):
    module_name, function_name = target.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


# ---------------------------------------------------------------------------
# Output format conversion
# ---------------------------------------------------------------------------

def tool_calls_to_messages(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert the agent's internal tool_calls list to the OpenAI-style message
    sequence expected by effolang-agent-evals' source="messages" extractor.

    Input shape (from ask()):
        [{"tool_name": str, "args": dict, "result": any, "id": str | None}]

    Output shape (OpenAI tool-call messages):
        [
          {"role": "assistant", "tool_calls": [...]},   # one per batch of calls
          {"role": "tool", "tool_call_id": ..., "content": ...},  # one per result
          ...
        ]
    """
    if not tool_calls:
        return []

    ai_tool_calls: list[dict] = []
    for i, tc in enumerate(tool_calls):
        name = tc.get("tool_name") or tc.get("name") or "unknown_tool"
        args = tc.get("args", {})
        call_id = tc.get("id") or f"call_{i}"
        ai_tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )

    messages: list[dict] = [{"role": "assistant", "tool_calls": ai_tool_calls}]
    for i, tc in enumerate(tool_calls):
        call_id = tc.get("id") or f"call_{i}"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(tc.get("result", "")),
            }
        )
    return messages


def expected_tools_to_trajectory(
    expected_tools: list[str],
    expected_tool_calls: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert golden dataset tool reference data into the OpenAI-style
    expected_trajectory format required by the reference-driven SDK evaluators.

    When expected_tool_calls is provided (list of {"name": str, "args": dict}),
    the real args are encoded — enabling tool_args_match to compare actual vs
    expected arguments. Falls back to empty args when only expected_tools is given.

    Order is preserved so make_reference_tool_sequence_evaluator works correctly.
    """
    if expected_tool_calls:
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"ref_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("args", {})),
                        },
                    }
                    for i, tc in enumerate(expected_tool_calls)
                ],
            }
        ]
    if not expected_tools:
        return []
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"ref_{i}",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }
                for i, name in enumerate(expected_tools)
            ],
        }
    ]


# ---------------------------------------------------------------------------
# LangSmith dataset helpers
# ---------------------------------------------------------------------------

def _sync_expected_trajectory(
    client: Any,
    dataset_name: str,
    rows: list[dict[str, Any]],
    question_field: str = "question",
    expected_tools_field: str = "expected_tools",
    expected_tool_calls_field: str = "expected_tool_calls",
) -> None:
    """
    For each existing LangSmith example, match it to a local golden row by
    question and rebuild expected_trajectory from the local reference data.

    Prefers expected_tool_calls (with real args) over expected_tools (names only).
    Falls back to whatever is stored in LangSmith when no local row matches.
    Idempotent.
    """
    local_by_question: dict[str, tuple[list[str], list[dict] | None]] = {}
    for row in rows:
        inputs = row.get("inputs", {})
        q = (inputs.get(question_field) or inputs.get("input") or inputs.get("question") or "").strip().lower()
        if q:
            outputs = row.get("outputs", {})
            local_by_question[q] = (
                outputs.get(expected_tools_field, []),
                outputs.get(expected_tool_calls_field),
            )

    existing_examples = list(client.list_examples(dataset_name=dataset_name))
    updated = 0
    for example in existing_examples:
        ex_inputs = example.inputs or {}
        q = (ex_inputs.get(question_field) or ex_inputs.get("input") or ex_inputs.get("question") or "").strip().lower()
        if q in local_by_question:
            expected_tools, expected_tool_calls = local_by_question[q]
        else:
            expected_tools = (example.outputs or {}).get(expected_tools_field, [])
            expected_tool_calls = None
        trajectory = expected_tools_to_trajectory(expected_tools, expected_tool_calls)
        if (example.outputs or {}).get("expected_trajectory") == trajectory:
            continue
        client.update_example(
            example.id,
            outputs={**(example.outputs or {}), "expected_trajectory": trajectory},
        )
        updated += 1
    print(
        f"[trajectory-evals] Synced expected_trajectory on "
        f"{updated}/{len(existing_examples)} examples."
    )


def ensure_dataset(
    client: Any,
    dataset_name: str,
    rows: list[dict[str, Any]],
    question_field: str = "question",
    expected_tools_field: str = "expected_tools",
    expected_tool_calls_field: str = "expected_tool_calls",
) -> str:
    """
    Push the golden rows to a LangSmith dataset, creating it if it does not
    exist.  Each row's expected_tool_calls (or expected_tools fallback) is
    promoted to expected_trajectory so the SDK evaluators work out of the box.

    Prefers expected_tool_calls when present — this enables tool_args_match
    to compare actual vs expected arguments. Falls back to expected_tools
    (names only, empty args) for rows without explicit call specs.

    Upload is idempotent: rows whose question already exists in LangSmith are
    skipped, so repeated CI runs never accumulate duplicate examples.

    Returns the dataset name (unchanged).
    """
    existing_datasets = list(client.list_datasets(dataset_name=dataset_name))
    if existing_datasets:
        dataset = existing_datasets[0]
        print(f"[trajectory-evals] Syncing existing LangSmith dataset: {dataset_name}")
        _sync_expected_trajectory(client, dataset_name, rows, question_field, expected_tools_field, expected_tool_calls_field)
    else:
        print(f"[trajectory-evals] Creating LangSmith dataset: {dataset_name}")
        dataset = client.create_dataset(dataset_name=dataset_name)

    # Dedup: only upload rows whose question is not already in the dataset.
    existing_questions = {
        (ex.inputs or {}).get(question_field, "").strip().lower()
        for ex in client.list_examples(dataset_name=dataset_name)
    }

    examples = []
    for row in rows:
        inputs = row.get("inputs", {})
        outputs = row.get("outputs", {})
        q = inputs.get(question_field, "").strip().lower()
        if q in existing_questions:
            continue
        expected_tools = outputs.get(expected_tools_field, [])
        expected_tool_calls = outputs.get(expected_tool_calls_field)
        examples.append(
            {
                "inputs": inputs,
                "outputs": {
                    **outputs,
                    "expected_trajectory": expected_tools_to_trajectory(expected_tools, expected_tool_calls),
                },
            }
        )

    if examples:
        client.create_examples(dataset_id=dataset.id, examples=examples)
        print(f"[trajectory-evals] Uploaded {len(examples)} new examples.")
    else:
        print(f"[trajectory-evals] All {len(rows)} examples already present — nothing to upload.")

    return dataset_name


# ---------------------------------------------------------------------------
# Agent runner wrapper
# ---------------------------------------------------------------------------

def make_runner(ask_fn, question_field: str = "question", context_field: str = "context"):
    """
    Return a callable compatible with langsmith.evaluate().

    langsmith.evaluate() passes `inputs` (the Example inputs dict) and
    expects a dict back.  We call ask(), convert tool_calls to messages, and
    return both so evaluators using source="messages" and those inspecting
    raw outputs both work.
    """
    def _run(inputs: dict) -> dict:
        question = (
            inputs.get(question_field)
            or inputs.get("input")
            or inputs.get("query")
            or ""
        )
        context = inputs.get(context_field) or inputs.get("context", {})
        response = ask_fn(question, **context)
        tool_calls: list[dict] = response.get("tool_calls", [])
        messages = tool_calls_to_messages(tool_calls)
        messages.append({"role": "assistant", "content": response.get("answer", "")})
        return {
            "messages": messages,
            "answer": response.get("answer", ""),
            "tool_calls": tool_calls,
        }

    return _run


# ---------------------------------------------------------------------------
# Evaluator plugin loader 
# ---------------------------------------------------------------------------
"""Dynamically load evaluator factories from config and return a list of
evaluator instances.  This allows users to specify which evaluators to run
without hardcoding them in the script, and supports passing args to them via the config file.

The config file should have a structure like:
trajectory_evaluators:
  source: messages  # optional default source for all evaluators
  reference_driven:
    - factory: make_reference_tool_sequence_evaluator
      name: "Expected tools in correct order"
      description: "Checks that the agent called the expected tools in the correct order."
      # Optional args to pass to the factory (besides `source`, which is set automatically)
      # tool_name: "some_tool"  # example of an arg that some factories might require
  hardcoded_invariants:
    - factory: make_no_undefined_tool_calls_evaluator
      name: "No undefined tool calls"
      description: "Checks that the agent did not call any tools that were not defined in the spec."
"""

def load_evaluators_from_config(config: dict) -> list:
    from effolang_agent_evals import trajectory as traj

    available = [n for n in dir(traj) if n.startswith("make_")]

    def _resolve_factory(name: str, section: str):
        fn = getattr(traj, name, None)
        if fn is None:
            raise ValueError(
                f"Unknown evaluator factory '{name}' in config section "
                f"'{section}'.\nAvailable factories: {available}"
            )
        return fn

    traj_cfg = config.get("trajectory_evaluators", {})
    default_source = traj_cfg.get("source", "messages")
    evaluators: list = []

    for spec in traj_cfg.get("reference_driven", []):
        factory_fn = _resolve_factory(spec["factory"], "reference_driven")
        kwargs = {k: v for k, v in spec.items() if k not in ("factory", "min")}
        kwargs.setdefault("source", default_source)
        evaluators.append(factory_fn(**kwargs))

    for spec in traj_cfg.get("hardcoded_invariants", []):
        factory_fn = _resolve_factory(spec["factory"], "hardcoded_invariants")
        kwargs = {k: v for k, v in spec.items() if k not in ("factory", "owner", "min")}
        kwargs.setdefault("source", default_source)
        evaluators.append(factory_fn(**kwargs))

    return evaluators


# ---------------------------------------------------------------------------
# LLM-as-judge evaluator loader
# ---------------------------------------------------------------------------

def load_llm_judges_from_config(config: dict) -> list:
    """Load LLM-as-judge evaluators from the llm_judges config block.

    Returns an empty list if the block is absent or enabled=false, so the
    caller can always merge the result without guarding.

    Requires effolang-agent-evals[llm] to be installed. If the block is
    enabled but the [llm] extra is missing, raises ImportError with a clear
    install hint rather than a cryptic AttributeError.
    """
    judges_cfg = config.get("llm_judges", {})
    if "llm_judges" not in config:
        print("[trajectory-evals] LLM judges disabled — skipping (llm_judges block absent).")
        return []
    if "enabled" not in judges_cfg:
        print(
            "[trajectory-evals] ERROR: llm_judges block present but 'enabled' key is missing. "
            "Set enabled: true or enabled: false explicitly."
        )
        sys.exit(1)
    if not judges_cfg["enabled"]:
        print("[trajectory-evals] LLM judges disabled — skipping.")
        return []

    try:
        from effolang_agent_evals import trajectory as traj
    except ImportError:
        raise ImportError(
            "[trajectory-evals] effolang-agent-evals is not installed.\n"
            "  Run: pip install 'effolang-agent-evals[llm]'"
        )

    available = [n for n in dir(traj) if n.startswith("make_")]
    model = judges_cfg.get("model", "openai:gpt-4o-mini")
    source = judges_cfg.get("source", "child_runs")
    aggregation = judges_cfg.get("aggregation", "mean")
    tool_descriptions = judges_cfg.get("tool_descriptions", {})

    evaluators = []
    for spec in judges_cfg.get("judges", []):
        factory_name = spec["factory"]
        fn = getattr(traj, factory_name, None)
        if fn is None:
            raise ValueError(
                f"Unknown LLM judge factory '{factory_name}' in llm_judges config.\n"
                f"Available factories: {available}"
            )
        kwargs: dict = {"key": spec["key"], "model": model, "source": source}
        # SDK uses different param names for the two per-call judges:
        # make_tool_selection_llm_judge_evaluator  → tool_descriptions
        # make_tool_invocation_llm_judge_evaluator → tool_schemas
        if factory_name == "make_tool_selection_llm_judge_evaluator":
            kwargs["tool_descriptions"] = tool_descriptions
            kwargs["aggregation"] = aggregation
        elif factory_name == "make_tool_invocation_llm_judge_evaluator":
            kwargs["tool_schemas"] = tool_descriptions
            kwargs["aggregation"] = aggregation
        evaluators.append(fn(**kwargs))
        print(f"[trajectory-evals] LLM judge loaded: {factory_name} (key={spec['key']})")

    return evaluators


# ---------------------------------------------------------------------------
# LangSmith experiment metadata (H4 — traceable scores)
# ---------------------------------------------------------------------------

def get_run_metadata(agent_name: str, config_path: str, dataset_path: str) -> dict[str, str]:
    """Return metadata dict for ls_evaluate() with evaluator version, agent SHA, dataset hash."""
    import hashlib
    import importlib.metadata
    import subprocess

    try:
        effo_version = importlib.metadata.version("effolang-agent-evals")
    except Exception:
        effo_version = "unknown"

    try:
        agent_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        agent_sha = "unknown"

    try:
        with open(dataset_path, "rb") as f:
            dataset_sha = hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        dataset_sha = "unknown"

    return {
        "agent_name": agent_name,
        "config_path": config_path,
        "golden_file": dataset_path,
        "evaluator_package": f"effolang-agent-evals=={effo_version}",
        "agent_sha": agent_sha,
        "dataset_sha": dataset_sha,
    }


# ---------------------------------------------------------------------------
# Per-evaluator threshold map
# ---------------------------------------------------------------------------

def build_threshold_map(config: dict, global_threshold: float) -> dict[str, float]:
    """
    Build a {feedback.<key>: min_score} map from eval.yaml.

    Each evaluator entry may carry an optional `min` field — the minimum
    average score that evaluator must achieve across all rows.  Entries
    without `min` fall back to the global required_pass_rate.

    Covers trajectory_evaluators.reference_driven,
    trajectory_evaluators.hardcoded_invariants, and llm_judges.judges.
    """
    thmap: dict[str, float] = {}
    traj_cfg = config.get("trajectory_evaluators", {})
    for section in ("reference_driven", "hardcoded_invariants"):
        for spec in traj_cfg.get(section, []):
            key = spec.get("key")
            if key:
                thmap[f"feedback.{key}"] = float(spec.get("min", global_threshold))
    for spec in config.get("llm_judges", {}).get("judges", []):
        key = spec.get("key")
        if key:
            thmap[f"feedback.{key}"] = float(spec.get("min", global_threshold))
    return thmap


# ---------------------------------------------------------------------------
# Threshold check (mirrors run_offline_evals.py exit-code logic)
# ---------------------------------------------------------------------------

def check_threshold(
    results: Any,
    required_pass_rate: float,
    threshold_map: "dict[str, float] | None" = None,
) -> bool:
    """
    Inspect the evaluate() ResultsDict and return True if every evaluator
    column meets its threshold.

    Per-evaluator thresholds are looked up from threshold_map
    (keyed as "feedback.<evaluator_key>").  Columns absent from the map
    fall back to required_pass_rate.  Returns True if results has no score
    data (nothing to gate on).
    """
    try:
        summary = results.to_pandas()
    except Exception:
        return True

    score_cols = [c for c in summary.columns if c.startswith("feedback.")]
    if not score_cols:
        return True

    any_scored = False
    passed = True
    for col in score_cols:
        col_scores = summary[col].dropna()
        if col_scores.empty:
            continue
        any_scored = True
        col_mean = col_scores.mean()
        threshold = (threshold_map or {}).get(col, required_pass_rate)
        status = "✅ PASS" if col_mean >= threshold else "❌ FAIL"
        print(
            f"[trajectory-evals] {status}: {col} score {col_mean:.2f} "
            f"(min {threshold:.2f})"
        )
        if col_mean < threshold:
            passed = False

    if not any_scored:
        # Every evaluator returned score=None (all rows were skipped).
        # This is not a passing state — it means expected_trajectory was
        # never written (upstream offline eval likely failed) or the dataset
        # is empty. Refuse to pass on zero real scores.
        print(
            f"[trajectory-evals] ERROR: {len(score_cols)} evaluator column(s) found "
            f"but every score was None (all rows skipped). "
            f"Upstream offline eval may have failed — refusing to pass on empty data."
        )
        return False

    return passed


# ---------------------------------------------------------------------------
# Report writer — always called after check_threshold, before sys.exit
# ---------------------------------------------------------------------------

def _write_reports(
    agent_name: str,
    results: Any,
    passed: bool,
    required_pass_rate: float,
) -> None:
    """Write eval_report_*.md and trajectory_eval_*.json to the working directory.

    Called regardless of pass/fail so failed runs still produce uploadable evidence.
    Errors are caught and printed rather than raised — a write failure must not
    shadow the real eval result.
    """
    import datetime as _dt
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = f"trajectory_eval_{agent_name}_{ts}.json"
    md_path = f"eval_report_{agent_name}_{ts}.md"

    try:
        df = results.to_pandas()
        records = df.to_dict(orient="records")
    except Exception:
        records = []

    payload = {
        "agent": agent_name,
        "passed": passed,
        "required_pass_rate": required_pass_rate,
        "experiment_url": getattr(results, "url", None),
        "rows": records,
    }
    try:
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"[trajectory-evals] Wrote {json_path}")
    except Exception as e:
        print(f"[trajectory-evals] WARNING: could not write {json_path}: {e}")

    status = "✅ PASSED" if passed else "❌ FAILED"
    md_lines = [
        f"# Trajectory Eval — {agent_name}",
        "",
        f"**Result:** {status}",
        f"**Required pass rate:** {required_pass_rate:.0%}",
        f"**Experiment:** {getattr(results, 'url', 'n/a')}",
    ]
    try:
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"[trajectory-evals] Wrote {md_path}")
    except Exception as e:
        print(f"[trajectory-evals] WARNING: could not write {md_path}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config_path: str) -> None:
    try:
        from langsmith import Client, evaluate as ls_evaluate
    except ImportError:
        print(
            "[trajectory-evals] ERROR: langsmith is not installed. "
            "Run: pip install langsmith"
        )
        sys.exit(1)

    try:
        import effolang_agent_evals  # noqa: F401 — presence check; factories loaded via loader
    except ImportError:
        print(
            "[trajectory-evals] ERROR: effolang-agent-evals is not installed. "
            "Run: pip install effolang-agent-evals"
        )
        sys.exit(1)

    config = load_yaml(config_path)
    agent_name: str = config["agent"]["name"]
    target: str = config["agent"]["target"]
    dataset_path: str = config["dataset"]["golden_file"]
    ls_dataset_name: str = config["dataset"].get(
        "langsmith_dataset", f"{agent_name}-trajectory-evals"
    )
    required_pass_rate: float = config.get("thresholds", {}).get(
        "required_pass_rate", 1.0  # MAP default — pending RAI approval
    )
    experiment_prefix: str = f"{agent_name}-trajectory"
    schema = config.get("dataset", {}).get("schema", {})
    question_field: str = schema.get("input_question", "question")
    context_field: str = schema.get("input_context", "context")
    expected_tools_field: str = schema.get("output_expected_tools", "expected_tools")

    print(f"[trajectory-evals] Agent:   {agent_name}")
    print(f"[trajectory-evals] Target:  {target}")
    print(f"[trajectory-evals] Dataset: {dataset_path}")

    ask_fn = load_target(target)
    golden_rows = load_jsonl(dataset_path)

    endpoint_env = (
        os.environ.get("LANGSMITH_ENDPOINT")
        or os.environ.get("LANGCHAIN_ENDPOINT")
        or "(not set — defaulting to LangSmith Cloud)"
    )
    print(f"[trajectory-evals] LANGSMITH_ENDPOINT: {endpoint_env}")

    client = Client()
    print(f"[trajectory-evals] Client API URL:     {client.api_url}")

    # Preflight: confirm the endpoint speaks JSON before making dataset calls.
    # The root URL returns HTML (the web UI); the API is always at /api.
    import requests as _requests
    try:
        r = _requests.get(
            f"{client.api_url}/info",
            headers={"x-api-key": os.environ.get("LANGSMITH_API_KEY", "")},
            timeout=10,
        )
        r.raise_for_status()
        info = r.json()
        print(f"[trajectory-evals] LangSmith version:  {info.get('version')}")
    except Exception as e:
        print(
            f"[trajectory-evals] ERROR: Preflight against {client.api_url}/info failed: {e}\n"
            f"  → Check that LANGSMITH_ENDPOINT points at the API (e.g. https://host/api),\n"
            f"    not the web UI root (https://host)."
        )
        sys.exit(1)

    try:
        ensure_dataset(client, ls_dataset_name, golden_rows, question_field, expected_tools_field)
    except Exception as e:
        print(
            f"[trajectory-evals] ERROR: LangSmith dataset sync failed.\n"
            f"  Endpoint: {client.api_url}\n"
            f"  Dataset:  {ls_dataset_name}\n"
            f"  Cause:    {type(e).__name__}: {e}"
        )
        sys.exit(1)

    evaluators = load_evaluators_from_config(config)
    evaluators += load_llm_judges_from_config(config)
    if not evaluators:
        print(
            "[trajectory-evals] ERROR: no evaluators loaded — "
            "trajectory_evaluators block is missing or empty in config. "
            "An eval that checks nothing cannot pass."
        )
        sys.exit(1)

    print(f"[trajectory-evals] Running evaluate() with {len(evaluators)} evaluators …")
    try:
        results = ls_evaluate(
            make_runner(ask_fn, question_field=question_field, context_field=context_field),
            data=ls_dataset_name,
            evaluators=evaluators,
            experiment_prefix=experiment_prefix,
            metadata=get_run_metadata(agent_name, config_path, dataset_path),
        )
    except Exception as e:
        print(
            f"[trajectory-evals] ERROR: ls_evaluate() failed.\n"
            f"  Endpoint: {client.api_url}\n"
            f"  Cause:    {type(e).__name__}: {e}"
        )
        _write_reports(agent_name, None, False, required_pass_rate)
        sys.exit(1)
    print(f"[trajectory-evals] Experiment URL: {results.url}")

    step_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary_path and results.url:
        with open(step_summary_path, "a") as f:
            f.write(f"\n**LangSmith experiment:** {results.url}\n")

    threshold_map = build_threshold_map(config, required_pass_rate)
    passed = check_threshold(results, required_pass_rate, threshold_map)
    _write_reports(agent_name, results, passed, required_pass_rate)  # always write before exit
    if not passed:
        sys.exit(1)

    print("[trajectory-evals] All evaluators passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trajectory evals via effolang-agent-evals + LangSmith."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to agent_eval_config.yaml",
    )
    args = parser.parse_args()
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    run(args.config)


if __name__ == "__main__":
    main()
