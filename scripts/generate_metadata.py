"""
Generate agents/{agent}/metadata.yml from deployment.yaml.

Usage: python scripts/generate_metadata.py <agent_name>

The `metadata` section of deployment.yaml is the single source of truth for 8451
platform metadata. The `component` field is always derived from `image_name` and
does not need to be set manually.
"""

import sys

import yaml


def build_yaml(component: str, metadata: dict) -> str:
    contacts = metadata.get("contacts", {})
    tags = metadata.get("tags", [])

    lines = [
        f"profit_stream: {metadata['profit_stream']}",
        f"product: {metadata['product']}",
        f"capability: {metadata['capability']}",
        f"component: {component}",
        f"business_domain: {metadata['business_domain']}",
        f"team: {metadata['team']}",
    ]

    if contacts:
        lines.append("contacts:")
        for contact_type, emails in contacts.items():
            lines.append(f"  {contact_type}:")
            for email in emails:
                lines.append(f"    - {email}")

    if tags:
        lines.append("tags:")
        for tag in tags:
            lines.append(f"  - {tag}")

    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_metadata.py <agent_name>", file=sys.stderr)
        sys.exit(1)

    agent_name = sys.argv[1]

    with open(f"agents/{agent_name}/deployment.yaml") as f:
        config = yaml.safe_load(f)

    image_name = config.get("image_name", agent_name.replace("_", "-"))
    metadata = config.get("metadata")

    if not metadata:
        print(f"Error: no 'metadata' section found in agents/{agent_name}/deployment.json", file=sys.stderr)
        sys.exit(1)

    output_path = f"agents/{agent_name}/metadata.yml"
    with open(output_path, "w") as f:
        f.write(build_yaml(image_name, metadata))

    print(f"Generated {output_path} (component: {image_name})")


if __name__ == "__main__":
    main()
