import asyncio
import sys
import uuid

from langgraph.types import Command

from app.domain.models.agent_state import AgentState
from app.graph import compile_graph


async def ask(question: str) -> str:
    graph = await compile_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = await graph.ainvoke(AgentState(question=question), config)

    while result.get("__interrupt__"):
        clarifying_question = result["__interrupt__"][0].value["question"]
        user_answer = input(f"{clarifying_question}\n> ")
        result = await graph.ainvoke(Command(resume=user_answer), config)

    return result["answer"]


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m app.main "your question"')
        sys.exit(1)

    question = sys.argv[1]
    answer = asyncio.run(ask(question))
    print(answer)


if __name__ == "__main__":
    main()
