"""
Run LangSmith evaluation against a deployed agent and report results.

Reads evaluation config from agents/{agent}/deployment.yaml:
  evaluation.dataset        - LangSmith dataset name to evaluate against
  evaluation.evaluator_model - model used for LLM-as-judge (default: gpt-4o-mini)
  evaluation.thresholds     - dict of metric -> expression (e.g. ">=0.8", "<=0.1")

If no "evaluation" key is present the script exits 0 (skips gracefully).

Required environment variables:
  LANGSMITH_API_KEY  - LangSmith API key
  LANGSMITH_ENDPOINT - e.g. https://langsmith.aks-ur-plg-internal.8451.cloud/api/v1

Optional:
  OPENAI_API_KEY     - needed for LLM-based evaluators
  OPENAI_API_BASE    - KrAIG proxy base URL (mapped to OPENAI_BASE_URL automatically)
"""

import argparse
import logging
import operator
import os
import sys
from collections import defaultdict

import requests
import yaml

# Map 8451 KrAIG proxy env var to the standard LangChain / OpenAI SDK var so
# that openevals and langchain_openai pick it up automatically.
if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Supported threshold operators, longest match first to avoid ">" shadowing ">="
OP_MAP = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

CORRECTNESS_PROMPT = """You are an expert evaluator grading an AI agent's response.

Input given to the agent:
{inputs}

Expected output (from evaluation dataset):
{reference_outputs}

Agent's actual output:
{outputs}

Score 1 if the agent's response correctly answers the input — key facts match even
if the wording differs. Score 0 if the response is incorrect, incomplete, or
significantly different from expected.

Respond with ONLY a JSON object: {{"score": 0 or 1, "reasoning": "brief explanation"}}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_threshold(expr) -> tuple:
    """Parse a threshold expression like '>=0.8' or bare float into (op_fn, value)."""
    s = str(expr).strip()
    for symbol in sorted(OP_MAP, key=len, reverse=True):
        if s.startswith(symbol):
            return OP_MAP[symbol], float(s[len(symbol):])
    # Bare float defaults to >=
    return operator.ge, float(s)


def load_config(agent: str) -> dict:
    path = f"agents/{agent}/deployment.yaml"
    if not os.path.exists(path):
        logger.error(f"deployment.yaml not found: {path}")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def get_langsmith_api_base() -> str:
    endpoint = os.environ["LANGSMITH_ENDPOINT"].rstrip("/")
    return endpoint.removesuffix("/api/v1") + "/api-host/v2"


def get_langgraph_url() -> str:
    endpoint = os.environ["LANGSMITH_ENDPOINT"].rstrip("/")
    return endpoint.removesuffix("/api/v1")


def find_deployment(deployment_name: str) -> dict | None:
    """Find a deployment by name, paging through all results until found or exhausted."""
    base = get_langsmith_api_base()
    headers = {"X-Api-Key": os.environ["LANGSMITH_API_KEY"]}

    # Add workspace/organization ID if available (LangSmith Cloud multi-tenant)
    workspace_id = os.getenv("LANGSMITH_WORKSPACE_ID")
    if workspace_id:
        headers["X-Tenant-Id"] = workspace_id

    url = f"{base}/deployments"
    params = {"limit": 100}

    logger.info(f"Searching for deployment: {deployment_name}")

    all_deployment_names = []

    while True:
        resp = requests.get(url, headers=headers, params=params)
        if not resp.ok:
            logger.error(f"Failed to list deployments — HTTP {resp.status_code}: {resp.text}")
            sys.exit(1)

        data = resp.json()

        # LangSmith API can return list, dict with "items", or dict with "resources"
        if isinstance(data, list):
            items = data
        else:
            items = data.get("resources") or data.get("items", [])

        if items:
            page_names = [d.get("name") for d in items if d.get("name")]
            all_deployment_names.extend(page_names)

        match = next((d for d in items if d.get("name") == deployment_name), None)
        if match:
            logger.info(f"Found deployment: {deployment_name} (ID: {match.get('id')})")
            return match

        # Handle cursor-based pagination — stop when no next cursor or empty page
        if not items or not isinstance(data, dict):
            break
        next_cursor = data.get("nextCursor") or data.get("next_cursor")
        if not next_cursor:
            break
        params = {"limit": 100, "cursor": next_cursor}

    logger.warning(f"Deployment '{deployment_name}' not found (searched {len(all_deployment_names)} deployments)")
    return None


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def build_correctness_evaluator(model: str):
    """
    Build an LLM-as-judge correctness evaluator following LangSmith conventions.
    Uses openevals.llm.create_llm_as_judge with the configured model.
    Returns None if OPENAI_API_KEY is not available.
    """
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set — skipping LLM-based correctness evaluator")
        return None

    try:
        from openevals.llm import create_llm_as_judge
    except ImportError:
        logger.warning("openevals not installed — falling back to langchain_openai evaluator")
        return _build_langchain_correctness_evaluator(model)

    logger.info(f"Building LLM-as-judge correctness evaluator (model: {model})")
    return create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        model=f"openai:{model}",
    )


def _build_langchain_correctness_evaluator(model: str):
    """Fallback evaluator using langchain_openai directly."""
    import json as _json

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        logger.warning("langchain_openai not installed — no LLM evaluator available")
        return None

    kwargs = {"model": model, "api_key": os.environ["OPENAI_API_KEY"]}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

    llm = ChatOpenAI(**kwargs)

    def correctness_evaluator(run, example):
        try:
            resp = llm.invoke(
                CORRECTNESS_PROMPT.format(
                    inputs=str(example.inputs or {}),
                    reference_outputs=str(example.outputs or {}),
                    outputs=str(run.outputs or {}),
                )
            )
            result = _json.loads(resp.content)
            return {
                "key": "correctness",
                "score": float(result["score"]),
                "comment": result.get("reasoning", ""),
            }
        except Exception as exc:
            logger.warning(f"Correctness evaluator failed for run {run.id}: {exc}")
            return {"key": "correctness", "score": None}

    return correctness_evaluator


# ---------------------------------------------------------------------------
# Threshold checking
# ---------------------------------------------------------------------------


def _iter_eval_results(eval_results):
    """Iterate over individual EvaluationResult objects regardless of SDK return type."""
    if not eval_results:
        return []
    if hasattr(eval_results, "results"):
        return eval_results.results or []
    if isinstance(eval_results, dict):
        return eval_results.get("results", [])
    return []


def check_thresholds(results: list, thresholds: dict) -> tuple[bool, dict[str, float]]:
    """
    Aggregate scores per metric key and compare against threshold expressions.
    Returns (all_passed, avg_scores_by_key).
    """
    scores: dict[str, list[float]] = defaultdict(list)
    for result in results:
        eval_results = result.get("evaluation_results")
        for eval_result in _iter_eval_results(eval_results):
            key = getattr(eval_result, "key", "") or ""
            score = getattr(eval_result, "score", None)
            if key and score is not None:
                scores[key].append(float(score))

    avgs: dict[str, float] = {}
    all_passed = True
    for metric, expr in thresholds.items():
        values = scores.get(metric, [])
        if not values:
            logger.warning(f"  {metric}: no results — skipping threshold check")
            continue
        avg = sum(values) / len(values)
        avgs[metric] = avg
        op_fn, threshold_val = parse_threshold(expr)
        passed = op_fn(avg, threshold_val)
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {metric}: {avg:.3f}  (threshold {expr})  [{status}]")
        if not passed:
            all_passed = False

    return all_passed, avgs


# ---------------------------------------------------------------------------
# Markdown report (for PR comments and step summaries)
# ---------------------------------------------------------------------------


def write_eval_report(
    agent: str,
    deployment_name: str,
    dataset_name: str,
    thresholds: dict,
    results: list,
    output_file: str,
):
    """Write a markdown evaluation report compatible with GitHub PR comments."""
    scores: dict[str, list[float]] = defaultdict(list)
    for result in results:
        eval_results = result.get("evaluation_results")
        for eval_result in _iter_eval_results(eval_results):
            key = getattr(eval_result, "key", "") or ""
            score = getattr(eval_result, "score", None)
            if key and score is not None:
                scores[key].append(float(score))

    rows = []
    num_passed = num_failed = 0
    for metric, expr in thresholds.items():
        values = scores.get(metric, [])
        avg = sum(values) / len(values) if values else None
        avg_str = f"{avg:.3f}" if avg is not None else "N/A"
        if avg is not None:
            op_fn, threshold_val = parse_threshold(expr)
            passed = op_fn(avg, threshold_val)
            status = "✅" if passed else "❌"
            if passed:
                num_passed += 1
            else:
                num_failed += 1
        else:
            status = "⚠️ N/A"
        rows.append((metric, avg_str, str(expr), status))

    with open(output_file, "w") as f:
        f.write(f"## 🧪 Evaluation Results — `{agent}`\n\n")
        f.write(f"**Deployment:** `{deployment_name}`  \n")
        f.write(f"**Dataset:** `{dataset_name}`\n\n")
        f.write("| Metric | Avg Score | Threshold | Pass? |\n")
        f.write("|--------|-----------|-----------|-------|\n")
        for row in rows:
            f.write(f"| {row[0]} | {row[1]} | `{row[2]}` | {row[3]} |\n")
        f.write("\n")
        if rows:
            f.write(f"**✅ {num_passed} passed, ❌ {num_failed} failed**\n")

    logger.info(f"Eval report written to {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate a deployed LangSmith agent")
    parser.add_argument("--agent", required=True, help="Agent directory name (e.g. langsmith_quickstart_agent)")
    parser.add_argument(
        "--deployment-name",
        help="Override the LangSmith deployment name (default: name from deployment.yaml)",
    )
    parser.add_argument(
        "--preview-pr",
        type=int,
        help="If set, evaluate the preview deployment for this PR number",
    )
    args = parser.parse_args()

    config = load_config(args.agent)
    eval_config = config.get("evaluation")

    if not eval_config:
        logger.info(f"No 'evaluation' config found for {args.agent} — skipping")
        sys.exit(0)

    dataset_name = eval_config["dataset"]
    thresholds = eval_config.get("thresholds", {})
    evaluator_model = eval_config.get("evaluator_model", "gpt-4o-mini")
    graph_name = config["graph_name"]

    # Resolve deployment name: explicit override > preview pattern > config default
    if args.deployment_name:
        deployment_name = args.deployment_name
    elif args.preview_pr:
        deployment_name = f"{config['name']}-pr-{args.preview_pr}"
    else:
        deployment_name = config["name"]

    logger.info(f"Evaluating deployment '{deployment_name}' against dataset '{dataset_name}'")
    logger.info(f"Thresholds: {thresholds}")

    deployment = find_deployment(deployment_name)
    is_preview = bool(args.preview_pr or args.deployment_name)

    if not deployment:
        # For preview evals the deployment must exist (it was just created in the same
        # run), so a missing deployment is a real failure. For production evals it is
        # expected on the very first deployment — skip gracefully so the agent can
        # bootstrap, and evals will run normally on every subsequent deploy.
        if is_preview:
            logger.error(
                f"Preview deployment '{deployment_name}' not found in LangSmith. "
                "The deploy step should have created it — check earlier job logs."
            )
            sys.exit(1)

        skip_reason = f"deployment `{deployment_name}` does not exist yet"
        skip_msg = "> Evals will run automatically on the next deployment.\n"
        logger.warning(
            f"Deployment '{deployment_name}' not found in LangSmith — "
            "skipping evaluation. This is expected on the first deployment of a new agent."
        )
    else:
        # Check that the deployment is fully ready before evaluating. A deployment
        # stuck in DEPLOYING (e.g. due to a CI timeout) will return 405 on every
        # agent call, scoring 0.000 and incorrectly blocking CI.
        deployment_status = (
            deployment.get("status")
            or deployment.get("latest_revision_status")
            or "UNKNOWN"
        )
        logger.info(f"Deployment '{deployment_name}' status: {deployment_status}")
        not_ready = deployment_status not in ("DEPLOYED", "UNKNOWN")
        if not_ready and not is_preview:
            skip_reason = f"deployment `{deployment_name}` is not yet ready (status: `{deployment_status}`)"
            skip_msg = "> Evals will run automatically once the deployment is fully provisioned.\n"
            deployment = None  # fall through to skip logic below
        elif not_ready and is_preview:
            logger.error(
                f"Preview deployment '{deployment_name}' is not ready (status: {deployment_status}). "
                "Check earlier job logs for deployment errors."
            )
            sys.exit(1)

    if not deployment:
        report_path = f"eval_comment_{args.agent}.md"
        with open(report_path, "w") as f:
            f.write(f"## 🧪 Evaluation Results — `{args.agent}`\n\n")
            f.write(f"> ⏭️ **Skipped** — {skip_reason}.  \n")
            f.write(skip_msg)
        step_summary = os.getenv("GITHUB_STEP_SUMMARY")
        if step_summary:
            with open(report_path) as src, open(step_summary, "a") as dst:
                dst.write(src.read())
                dst.write("\n")
        sys.exit(0)

    try:
        from langsmith import Client, evaluate
        from langgraph.pregel.remote import RemoteGraph
    except ImportError as e:
        logger.error(f"Missing dependency: {e}. Ensure langsmith and langgraph are installed.")
        sys.exit(1)

    api_key = os.environ["LANGSMITH_API_KEY"]
    langgraph_url = get_langgraph_url()

    remote_agent = RemoteGraph(graph_name, url=langgraph_url, api_key=api_key)

    def invoke_agent(inputs: dict) -> dict:
        """Adapter: wraps RemoteGraph.invoke for use as an evaluate() target."""
        return remote_agent.invoke(inputs)

    client = Client(api_url=os.environ["LANGSMITH_ENDPOINT"], api_key=api_key)

    # Build evaluator list — gracefully skips if OPENAI_API_KEY is not available
    evaluators = []
    correctness_evaluator = build_correctness_evaluator(evaluator_model)
    if correctness_evaluator:
        evaluators.append(correctness_evaluator)
        logger.info(f"Using LLM-as-judge correctness evaluator (model: {evaluator_model})")
    else:
        logger.warning("No evaluators configured — threshold checks will find no scores")

    results = evaluate(
        invoke_agent,
        data=dataset_name,
        experiment_prefix=f"{deployment_name}-ci",
        client=client,
        evaluators=evaluators,
        project_name=graph_name,
    )

    result_list = list(results)

    logger.info("Evaluation complete. Checking metrics against thresholds:")
    passed, _ = check_thresholds(result_list, thresholds)

    # Write markdown report for PR comment / step summary
    report_path = f"eval_comment_{args.agent}.md"
    write_eval_report(
        agent=args.agent,
        deployment_name=deployment_name,
        dataset_name=dataset_name,
        thresholds=thresholds,
        results=result_list,
        output_file=report_path,
    )

    # Append to GitHub Actions step summary if running in CI
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(report_path) as src, open(step_summary, "a") as dst:
            dst.write(src.read())
            dst.write("\n")

    if not passed:
        logger.error("Evaluation FAILED — one or more metrics are below threshold. Deployment blocked.")
        sys.exit(1)

    logger.info("Evaluation PASSED — all metrics meet thresholds.")


if __name__ == "__main__":
    main()