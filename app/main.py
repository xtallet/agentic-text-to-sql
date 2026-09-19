import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

from app.domain.models.agent_state import AgentState
from app.graph import compile_graph
from app.infrastructure.services.langsmith import build_langsmith_config

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


async def ask(question: str) -> str:
    graph = await compile_graph()
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        **build_langsmith_config(question),
    }

    result = await graph.ainvoke(AgentState(question=question), config)

    while result.get("__interrupt__"):
        clarifying_question = result["__interrupt__"][0].value["question"]
        user_answer = input(f"{clarifying_question}\n> ")
        result = await graph.ainvoke(Command(resume=user_answer), config)

    answer = result["answer"]
    confidence = result.get("confidence")
    if confidence:
        answer += (
            f"\n\nConfidence: {confidence.capitalize()} ({result['confidence_reason']})"
        )

    return answer


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m app.main "your question"')
        sys.exit(1)

    question = sys.argv[1]
    answer = asyncio.run(ask(question))
    print(answer)


if __name__ == "__main__":
    main()
