import os
from langchain_openai import ChatOpenAI


def get_llm(model: str | None = None, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "claude-4.6-sonnet"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        **kwargs,
    )
