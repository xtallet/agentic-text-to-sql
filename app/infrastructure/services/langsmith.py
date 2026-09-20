from typing import Any

from app.config.settings import get_settings


def build_langsmith_config(question: str) -> dict[str, Any]:
    return {
        "metadata": {
            "question": question,
            "llm_model": get_settings().OPENAI_MODEL,
        },
        "tags": ["agentic-text-to-sql", "cli"],
        "run_name": f"🎵 {question[:60]}",
    }
