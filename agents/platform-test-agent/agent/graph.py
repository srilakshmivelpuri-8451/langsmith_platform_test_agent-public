"""LLM-backed platform-test-agent.

Built to prove the MAP cross-repo deploy_agent.yml pipeline (lint, offline
eval, trajectory eval, deploy) works end-to-end for a repo outside
langsmith-agent-quickstart. Uses a real LLM call (agents._shared.llm.get_llm,
routed through the internal OpenAI-compatible proxy) with a single tool, so
the deploy also proves the LLM secrets/proxy wiring, not just a rule-based
router.
"""
from langchain.agents import create_agent
from langchain_core.tools import tool

from agents._shared.llm import get_llm

# Canned, deterministic tool output. Each entry has one distinctive keyword
# so expected_answer_contains checks stay reliable even though the LLM
# composes the surrounding sentence.
_STATUS = {
    "pipeline": "Pipeline status: OPERATIONAL. Lint, offline-eval, trajectory-eval, and deploy stages are all green.",
    "deployment": "Deployment status: DEPLOYED. The LangSmith revision is live and serving traffic.",
    "runner": "Runner status: ONLINE. The self-hosted ephemeral runner is available and accepting jobs.",
    "langsmith": "LangSmith status: CONNECTED. The API is reachable and authenticated.",
}
COMPONENTS = list(_STATUS)


@tool
def check_platform_status(component: str) -> str:
    """Return the current status of a MAP platform component.

    Args:
        component: one of "pipeline", "deployment", "runner", "langsmith".
    """
    return _STATUS.get(
        component.lower().strip(),
        f"Unknown component '{component}'. Known components: {', '.join(COMPONENTS)}.",
    )


TOOLS = [check_platform_status]

SYSTEM_PROMPT = (
    "You are platform-test-agent, a diagnostic assistant for the MAP "
    "cross-repo deployment pipeline. The only components you can report on "
    "are: pipeline, deployment, runner, langsmith. "
    "If the user asks about the status of one of those components, call "
    "check_platform_status with that exact component name and reply with "
    "the tool's returned text verbatim, no extra commentary. "
    "If the user asks about any other component or topic, do not call the "
    "tool — reply in one short sentence that you only report on pipeline, "
    "deployment, runner, and langsmith status. "
    "For greetings or anything unrelated to platform status, reply briefly "
    "and conversationally without calling any tool."
)

GRAPH = create_agent(
    model=get_llm(temperature=0),
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)
