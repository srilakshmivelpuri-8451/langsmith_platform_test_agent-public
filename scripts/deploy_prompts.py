import argparse
import glob
import logging
import os
import sys
import yaml
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        default="",
        help="Only push this agent's prompts (agents/<agent>/prompts/*.yaml). "
             "Omit to push every agent's prompts (fleet-wide deploy).",
    )
    args = parser.parse_args()

    glob_pattern = f"agents/{args.agent}/prompts/*.yaml" if args.agent else "agents/*/prompts/*.yaml"
    client = Client()
    prompt_files = sorted(glob.glob(glob_pattern))

    if not prompt_files:
        if args.agent:
            sys.exit(f"No prompt files found for agent '{args.agent}' under {glob_pattern}")
        logger.warning(f"No prompt files found under {glob_pattern}")
        return

    pushed = 0
    failed = []
    for path in prompt_files:
        agent_name = path.split(os.sep)[1]
        with open(path) as f:
            prompt = yaml.safe_load(f)

        # Wrap template string in ChatPromptTemplate for LangSmith Hub
        prompt_object = ChatPromptTemplate.from_messages([
            ("system", prompt["template"])
        ])

        try:
            client.push_prompt(
                prompt["name"],
                object=prompt_object,
                description=prompt.get("description", ""),
            )
            logger.info(f"[{agent_name}] pushed prompt: {prompt['name']}")
            pushed += 1
        except Exception as e:
            # 409 Conflict means prompt unchanged - this is fine
            if "409" in str(e) or "Conflict" in str(e):
                logger.info(f"[{agent_name}] prompt {prompt['name']} is up-to-date (no changes)")
                pushed += 1
            else:
                logger.error(f"[{agent_name}] failed to push prompt {prompt['name']}: {e}")
                failed.append(f"{agent_name}/{prompt['name']}")

    logger.info(f"Deployed {pushed}/{len(prompt_files)} prompt(s) across {glob_pattern}")

    if failed:
        sys.exit(f"Failed to push {len(failed)} prompt(s): {failed}")

#TODO setup evals

if __name__ == "__main__":
    main()
