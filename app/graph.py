from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.config.logger import setup_logging
from app.domain.models.agent_state import AgentState
from app.domain.prompts.sql_generation import (
    ANSWER_SYSTEM_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    build_sql_system_prompt,
)
from app.infrastructure.di.dependencies import get_llm_adapter, get_sql_executor

logger = setup_logging()(__name__)

MAX_SQL_RETRIES = 2


class SqlQuery(BaseModel):
    query: str


class SqlEvaluation(BaseModel):
    is_valid: bool
    reason: str


async def generate_sql(state: AgentState) -> AgentState:
    sql_executor = get_sql_executor()
    if state.schema_description is None:
        state.schema_description = sql_executor.get_schema()

    llm = get_llm_adapter().get_llm_client()
    structured_llm = llm.with_structured_output(SqlQuery)

    if state.sql_error:
        state.retry_count += 1
        state.failed_attempts.append(
            f"Attempt {len(state.failed_attempts) + 1}:\n"
            f"SQL: {state.sql_query}\n"
            f"Error: {state.sql_error}"
        )

    human_content = state.question
    if state.failed_attempts:
        history = "\n\n".join(state.failed_attempts)
        human_content = (
            f"{state.question}\n\n"
            f"Previous failed attempts:\n{history}\n\n"
            "Write a corrected query that answers the question and avoids all of "
            "the errors above."
        )

    messages = [
        SystemMessage(content=build_sql_system_prompt(state.schema_description)),
        HumanMessage(content=human_content),
    ]

    state.sql_error = None
    state.sql_result = None

    try:
        result: SqlQuery = await structured_llm.ainvoke(messages)
        state.sql_query = result.query
    except Exception as e:
        logger.exception(f"Failed to generate SQL for question '{state.question}'")
        state.sql_error = str(e)

    return state


async def execute_sql(state: AgentState) -> AgentState:
    if state.sql_error or not state.sql_query:
        return state

    sql_executor = get_sql_executor()
    try:
        state.sql_result = sql_executor.execute(state.sql_query)
    except Exception as e:
        logger.warning(f"Failed to execute SQL '{state.sql_query}': {e}")
        state.sql_error = str(e)

    return state


async def evaluate_sql_result(state: AgentState) -> AgentState:
    llm = get_llm_adapter().get_llm_client()
    structured_llm = llm.with_structured_output(SqlEvaluation)

    messages = [
        SystemMessage(content=EVALUATION_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question: {state.question}\n"
                f"SQL: {state.sql_query}\n"
                f"Result:\n{state.sql_result}"
            )
        ),
    ]

    try:
        evaluation: SqlEvaluation = await structured_llm.ainvoke(messages)
        if not evaluation.is_valid:
            state.sql_error = (
                f"Self-evaluation rejected the result: {evaluation.reason}"
            )
    except Exception:
        # Fail open: an evaluator error shouldn't block an otherwise valid result.
        logger.exception(
            f"Failed to self-evaluate SQL result for question '{state.question}'"
        )

    return state


async def generate_answer(state: AgentState) -> AgentState:
    if state.sql_error:
        human_content = f"Question: {state.question}\nThe query failed with error: {state.sql_error}"
    else:
        human_content = f"Question: {state.question}\nSQL: {state.sql_query}\nResult:\n{state.sql_result}"

    llm = get_llm_adapter().get_llm_client()
    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]

    try:
        result = await llm.ainvoke(messages)
        state.answer = result.content
    except Exception:
        logger.exception(f"Failed to generate answer for question '{state.question}'")
        state.answer = "Sorry, I couldn't generate an answer due to an internal error."

    return state


def _route_on_error(state: AgentState) -> str:
    if state.retry_count < MAX_SQL_RETRIES:
        return "generate_sql"
    return "generate_answer"


def route_after_execute(state: AgentState) -> str:
    if state.sql_error:
        return _route_on_error(state)
    return "evaluate_sql_result"


def route_after_evaluate(state: AgentState) -> str:
    if state.sql_error:
        return _route_on_error(state)
    return "generate_answer"


async def compile_graph():
    agent_graph = StateGraph(AgentState)

    agent_graph.add_node("generate_sql", generate_sql)
    agent_graph.add_node("execute_sql", execute_sql)
    agent_graph.add_node("evaluate_sql_result", evaluate_sql_result)
    agent_graph.add_node("generate_answer", generate_answer)

    agent_graph.add_edge(START, "generate_sql")
    agent_graph.add_edge("generate_sql", "execute_sql")
    agent_graph.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {
            "generate_sql": "generate_sql",
            "evaluate_sql_result": "evaluate_sql_result",
            "generate_answer": "generate_answer",
        },
    )
    agent_graph.add_conditional_edges(
        "evaluate_sql_result",
        route_after_evaluate,
        {"generate_sql": "generate_sql", "generate_answer": "generate_answer"},
    )
    agent_graph.add_edge("generate_answer", END)

    return agent_graph.compile()
