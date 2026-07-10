"""
Create or update a LangSmith deployment for a specific agent and poll until it completes.

Reads deployment config from agents/{agent}/deployment.yaml. If the deployment
already exists it is updated (PATCH); otherwise a new one is created (POST).
Deployment lookup is always done by name — no external state file is required.

Required environment variables:
  LANGSMITH_API_KEY  - LangSmith API key
  LANGSMITH_ENDPOINT - e.g. https://langsmith.aks-ur-plg-internal.8451.cloud/api/v1
  LANGSMITH_IMAGE_URI - Full Artifactory image URI including tag

Any env var listed in deployment.yaml "secrets" must also be set.

Optional environment variables:
  LANGSMITH_LISTENER_ID - Listener ID for the external_docker deployment
"""

import argparse
import logging
import os
import sys
import time

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 20
MAX_POLL_ATTEMPTS = 30
DEFAULT_RESOURCE_SPEC = {
    "min_scale": 1,
    "max_scale": 1,
    "cpu": 0.5,
    "memory_mb": 512,
}


def get_base_url() -> str:
    endpoint = os.environ["LANGSMITH_ENDPOINT"].rstrip("/")
    # LANGSMITH_ENDPOINT is .../api/v1 — the deployment API lives on /api-host/v2
    host = endpoint.removesuffix("/api/v1")
    return f"{host}/api-host/v2"


def load_config(agent: str) -> dict:
    config_path = f"agents/{agent}/deployment.yaml"
    if not os.path.exists(config_path):
        logger.error(f"deployment.yaml not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_secrets(config: dict) -> list[dict]:
    secrets = []

    # Sensitive values — read from the runner environment (GitHub secrets/vars)
    for name in config.get("secrets", []):
        value = os.getenv(name)
        if value is None:
            logger.warning(f"Secret '{name}' listed in deployment.yaml but not set in environment — skipping")
            continue
        secrets.append({"name": name, "value": value})

    # Non-sensitive values — defined directly in deployment.yaml
    for name, value in config.get("env", {}).items():
        secrets.append({"name": name, "value": value})

    if secrets:
        logger.info(f"Syncing {len(secrets)} secret(s): {[s['name'] for s in secrets]}")
    return secrets


def find_deployment(base_url: str, headers: dict, name: str) -> dict | None:
    """Find a deployment by name, paging through all results until found or exhausted."""
    url = f"{base_url}/deployments"
    params: dict = {"limit": 100}
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

        match = next((d for d in items if d.get("name") == name), None)
        if match:
            logger.info(f"Found existing deployment: {name} (ID: {match.get('id')})")
            return match

        # Handle cursor-based pagination — stop when no next cursor or empty page
        if not items or not isinstance(data, dict):
            break
        next_cursor = data.get("nextCursor") or data.get("next_cursor")
        if not next_cursor:
            break
        params = {"limit": 100, "cursor": next_cursor}

    return None


def create_deployment(
    base_url: str,
    headers: dict,
    name: str,
    image_uri: str,
    secrets: list[dict],
    resource_spec: dict,
    listener_id: str | None,
) -> tuple[str, str]:
    logger.info(f"No existing deployment found — creating: {name}")
    source_config: dict = {"resource_spec": resource_spec}
    if listener_id:
        source_config["listener_id"] = listener_id

    resp = requests.post(
        f"{base_url}/deployments",
        headers=headers,
        json={
            "name": name,
            "source": "external_docker",
            "source_config": source_config,
            "source_revision_config": {"image_uri": image_uri},
            "secrets": secrets,
        },
    )

    if not resp.ok:
        # If deployment already exists (409 Conflict), try to find and update it
        if resp.status_code == 409 or "already exists" in resp.text.lower():
            logger.warning(f"Deployment {name} already exists (409) — retrying as update")
            existing = find_deployment(base_url, headers, name)
            if existing:
                return update_deployment(base_url, headers, existing["id"], image_uri, secrets)
        logger.error(f"Deployment creation failed — HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    deployment_id = data.get("id", "")
    revision_id = data.get("latest_revision_id", "")
    logger.info(f"Deployment ID: {deployment_id}")
    return deployment_id, revision_id


def update_deployment(
    base_url: str,
    headers: dict,
    deployment_id: str,
    image_uri: str,
    secrets: list[dict],
) -> tuple[str, str]:
    logger.info(f"Existing deployment found — updating image: {image_uri}")
    resp = requests.patch(
        f"{base_url}/deployments/{deployment_id}",
        headers=headers,
        json={
            "source_revision_config": {"image_uri": image_uri},
            "secrets": secrets,
        },
    )

    if not resp.ok:
        logger.error(f"Deployment update failed — HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    revision_id = data.get("revision_id") or data.get("latest_revision_id", "")
    logger.info(f"Revision ID: {revision_id}")
    return deployment_id, revision_id


def poll_revision(base_url: str, headers: dict, deployment_id: str, revision_id: str):
    if not revision_id:
        logger.info("No revision_id in response — check LangSmith for status.")
        sys.exit(0)

    logger.info(f"Polling for completion (up to {MAX_POLL_ATTEMPTS} attempts, {POLL_INTERVAL_SECONDS}s apart)...")
    for attempt in range(1, MAX_POLL_ATTEMPTS + 1):
        time.sleep(POLL_INTERVAL_SECONDS)
        resp = requests.get(
            f"{base_url}/deployments/{deployment_id}/revisions/{revision_id}",
            headers=headers,
        )
        status = resp.json().get("status", "UNKNOWN")
        logger.info(f"Attempt {attempt}/{MAX_POLL_ATTEMPTS}: status = {status}")

        if status == "DEPLOYED":
            logger.info("Deployment succeeded.")
            return
        if status in ("FAILED", "ERROR"):
            logger.error(f"Deployment failed with status: {status}")
            sys.exit(1)

    logger.error("Timed out waiting for deployment to complete. Check LangSmith manually.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Deploy a LangSmith agent")
    parser.add_argument("--agent", required=True, help="Agent directory name (e.g. announcement_agent)")
    parser.add_argument(
        "--deployment-name",
        help="Override the LangSmith deployment name (default: name from deployment.yaml)",
    )
    parser.add_argument(
        "--preview-pr",
        type=int,
        help="Deploy as a preview for this PR number (names deployment {name}-pr-{N})",
    )
    args = parser.parse_args()

    config = load_config(args.agent)
    if args.deployment_name:
        name = args.deployment_name
    elif args.preview_pr:
        name = f"{config['name']}-pr-{args.preview_pr}"
    else:
        name = config["name"]

    image_uri = os.environ["LANGSMITH_IMAGE_URI"]
    resource_spec = {**DEFAULT_RESOURCE_SPEC, **config.get("resource_spec", {})}
    listener_id = os.getenv("LANGSMITH_LISTENER_ID")
    secrets = build_secrets(config)

    headers = {
        "X-Api-Key": os.environ["LANGSMITH_API_KEY"],
        "Content-Type": "application/json",
    }

    # Add workspace/organization ID if available (LangSmith Cloud multi-tenant)
    workspace_id = os.getenv("LANGSMITH_WORKSPACE_ID")
    if workspace_id:
        headers["X-Tenant-Id"] = workspace_id

    base_url = get_base_url()

    existing = find_deployment(base_url, headers, name)

    if existing:
        deployment_id, revision_id = update_deployment(base_url, headers, existing["id"], image_uri, secrets)
    else:
        deployment_id, revision_id = create_deployment(
            base_url, headers, name, image_uri, secrets, resource_spec, listener_id
        )

    poll_revision(base_url, headers, deployment_id, revision_id)


if __name__ == "__main__":
    main()
