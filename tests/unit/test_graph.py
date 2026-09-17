from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph import END, START

from app.domain.exceptions.sql_exceptions import UnsafeSqlError
from app.domain.models.agent_state import AgentState
from app.graph import (
    SqlQuery,
    compile_graph,
    execute_sql,
    generate_answer,
    generate_sql,
)


class TestGenerateSql:
    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    @patch("app.graph.get_sql_executor")
    async def test_success(self, mock_get_sql_executor, mock_get_llm_adapter):
        mock_get_sql_executor.return_value.get_schema.return_value = (
            "CREATE TABLE Artist (...)"
        )

        structured_llm = AsyncMock()
        structured_llm.ainvoke.return_value = SqlQuery(query="SELECT * FROM Artist")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured_llm
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(question="How many artists are there?")
        result = await generate_sql(state)

        assert result.sql_query == "SELECT * FROM Artist"
        assert result.schema_description == "CREATE TABLE Artist (...)"
        assert result.sql_error is None

    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    @patch("app.graph.get_sql_executor")
    async def test_llm_failure_is_captured_as_error(
        self, mock_get_sql_executor, mock_get_llm_adapter
    ):
        mock_get_sql_executor.return_value.get_schema.return_value = (
            "CREATE TABLE Artist (...)"
        )

        structured_llm = AsyncMock()
        structured_llm.ainvoke.side_effect = RuntimeError("LLM is down")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured_llm
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(question="How many artists are there?")
        result = await generate_sql(state)

        assert result.sql_query is None
        assert result.sql_error == "LLM is down"

    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    @patch("app.graph.get_sql_executor")
    async def test_reuses_existing_schema_description(
        self, mock_get_sql_executor, mock_get_llm_adapter
    ):
        structured_llm = AsyncMock()
        structured_llm.ainvoke.return_value = SqlQuery(query="SELECT * FROM Artist")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured_llm
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(
            question="How many artists are there?",
            schema_description="cached schema",
        )
        await generate_sql(state)

        mock_get_sql_executor.return_value.get_schema.assert_not_called()


class TestExecuteSql:
    @pytest.mark.asyncio
    @patch("app.graph.get_sql_executor")
    async def test_success(self, mock_get_sql_executor):
        mock_get_sql_executor.return_value.execute.return_value = "id, name\n1, AC/DC"

        state = AgentState(question="q", sql_query="SELECT * FROM Artist")
        result = await execute_sql(state)

        assert result.sql_result == "id, name\n1, AC/DC"
        assert result.sql_error is None

    @pytest.mark.asyncio
    @patch("app.graph.get_sql_executor")
    async def test_unsafe_query_is_captured_as_error(self, mock_get_sql_executor):
        mock_get_sql_executor.return_value.execute.side_effect = UnsafeSqlError(
            "Only a single read-only SELECT statement is allowed."
        )

        state = AgentState(question="q", sql_query="DROP TABLE Artist")
        result = await execute_sql(state)

        assert result.sql_result is None
        assert "SELECT" in result.sql_error

    @pytest.mark.asyncio
    @patch("app.graph.get_sql_executor")
    async def test_skips_execution_if_sql_generation_already_failed(
        self, mock_get_sql_executor
    ):
        state = AgentState(question="q", sql_error="could not generate SQL")
        result = await execute_sql(state)

        mock_get_sql_executor.return_value.execute.assert_not_called()
        assert result.sql_error == "could not generate SQL"


class TestGenerateAnswer:
    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_success(self, mock_get_llm_adapter):
        llm = AsyncMock()
        llm.ainvoke.return_value = MagicMock(content="AC/DC has the most tracks.")
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(question="q", sql_query="SELECT 1", sql_result="id\n1")
        result = await generate_answer(state)

        assert result.answer == "AC/DC has the most tracks."

    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_llm_failure_returns_graceful_message(self, mock_get_llm_adapter):
        llm = AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("LLM is down")
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(question="q", sql_query="SELECT 1", sql_result="id\n1")
        result = await generate_answer(state)

        assert "couldn't generate an answer" in result.answer


class TestCompileGraph:
    @pytest.mark.asyncio
    @patch("app.graph.StateGraph")
    async def test_compile_graph_wires_nodes_and_edges(self, mock_state_graph):
        mock_graph = MagicMock()
        mock_state_graph.return_value = mock_graph
        mock_compiled_graph = MagicMock()
        mock_graph.compile.return_value = mock_compiled_graph

        result = await compile_graph()

        mock_state_graph.assert_called_once_with(AgentState)

        expected_nodes = [
            ("generate_sql", generate_sql),
            ("execute_sql", execute_sql),
            ("generate_answer", generate_answer),
        ]
        for name, node in expected_nodes:
            mock_graph.add_node.assert_any_call(name, node)
        assert mock_graph.add_node.call_count == len(expected_nodes)

        mock_graph.add_edge.assert_any_call(START, "generate_sql")
        mock_graph.add_edge.assert_any_call("generate_sql", "execute_sql")
        mock_graph.add_edge.assert_any_call("execute_sql", "generate_answer")
        mock_graph.add_edge.assert_any_call("generate_answer", END)

        mock_graph.compile.assert_called_once()
        assert result == mock_compiled_graph
