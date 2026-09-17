import asyncio
import sys

from app.domain.models.agent_state import AgentState
from app.graph import compile_graph


async def ask(question: str) -> str:
    graph = await compile_graph()
    result = await graph.ainvoke(AgentState(question=question))
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
