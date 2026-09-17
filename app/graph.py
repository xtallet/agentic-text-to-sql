from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.config.logger import setup_logging
from app.domain.models.agent_state import AgentState
from app.domain.prompts.sql_generation import (
    ANSWER_SYSTEM_PROMPT,
    build_sql_system_prompt,
)
from app.infrastructure.di.dependencies import get_llm_adapter, get_sql_executor

logger = setup_logging()(__name__)


class SqlQuery(BaseModel):
    query: str


async def generate_sql(state: AgentState) -> AgentState:
    sql_executor = get_sql_executor()
    if state.schema_description is None:
        state.schema_description = sql_executor.get_schema()

    llm = get_llm_adapter().get_llm_client()
    structured_llm = llm.with_structured_output(SqlQuery)

    messages = [
        SystemMessage(content=build_sql_system_prompt(state.schema_description)),
        HumanMessage(content=state.question),
    ]

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


async def compile_graph():
    agent_graph = StateGraph(AgentState)

    agent_graph.add_node("generate_sql", generate_sql)
    agent_graph.add_node("execute_sql", execute_sql)
    agent_graph.add_node("generate_answer", generate_answer)

    agent_graph.add_edge(START, "generate_sql")
    agent_graph.add_edge("generate_sql", "execute_sql")
    agent_graph.add_edge("execute_sql", "generate_answer")
    agent_graph.add_edge("generate_answer", END)

    return agent_graph.compile()
