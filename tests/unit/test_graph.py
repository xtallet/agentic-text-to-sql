from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.graph import END, START

from app.domain.exceptions.sql_exceptions import UnsafeSqlError
from app.domain.models.agent_state import AgentState
from app.graph import (
    MAX_SQL_RETRIES,
    SqlEvaluation,
    SqlQuery,
    compile_graph,
    evaluate_sql_result,
    execute_sql,
    generate_answer,
    generate_sql,
    route_after_evaluate,
    route_after_execute,
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

    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    @patch("app.graph.get_sql_executor")
    async def test_retry_includes_previous_error_and_clears_it(
        self, mock_get_sql_executor, mock_get_llm_adapter
    ):
        structured_llm = AsyncMock()
        structured_llm.ainvoke.return_value = SqlQuery(
            query="SELECT * FROM Artist LIMIT 1"
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured_llm
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(
            question="How many artists are there?",
            schema_description="cached schema",
            sql_query="SELECT * FROM Artsit",
            sql_error="no such table: Artsit",
            sql_result=None,
        )
        result = await generate_sql(state)

        human_message = structured_llm.ainvoke.call_args[0][0][1]
        assert "SELECT * FROM Artsit" in human_message.content
        assert "no such table: Artsit" in human_message.content
        assert result.sql_query == "SELECT * FROM Artist LIMIT 1"
        assert result.sql_error is None
        assert result.failed_attempts == [
            "Attempt 1:\nSQL: SELECT * FROM Artsit\nError: no such table: Artsit"
        ]
        assert result.retry_count == 1

    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    @patch("app.graph.get_sql_executor")
    async def test_second_retry_accumulates_both_failed_attempts(
        self, mock_get_sql_executor, mock_get_llm_adapter
    ):
        structured_llm = AsyncMock()
        structured_llm.ainvoke.return_value = SqlQuery(query="SELECT 1")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured_llm
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(
            question="How many artists are there?",
            schema_description="cached schema",
            sql_query="SELECT COUNT(*) FROM Artist WHERE",
            sql_error="syntax error",
            retry_count=1,
            failed_attempts=[
                "Attempt 1:\nSQL: SELECT * FROM Artsit\nError: no such table: Artsit"
            ],
        )
        result = await generate_sql(state)

        human_message = structured_llm.ainvoke.call_args[0][0][1]
        assert "Attempt 1" in human_message.content
        assert "no such table: Artsit" in human_message.content
        assert "Attempt 2" in human_message.content
        assert "syntax error" in human_message.content
        assert len(result.failed_attempts) == 2
        assert result.retry_count == 2


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


class TestEvaluateSqlResult:
    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_valid_result_leaves_sql_error_unset(self, mock_get_llm_adapter):
        structured_llm = AsyncMock()
        structured_llm.ainvoke.return_value = SqlEvaluation(
            is_valid=True, reason="Matches the question"
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured_llm
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(
            question="How many artists are there?",
            sql_query="SELECT COUNT(*) FROM Artist",
            sql_result="275",
        )
        result = await evaluate_sql_result(state)

        assert result.sql_error is None

    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_invalid_result_sets_sql_error_with_reason(
        self, mock_get_llm_adapter
    ):
        structured_llm = AsyncMock()
        structured_llm.ainvoke.return_value = SqlEvaluation(
            is_valid=False, reason="Result is empty but artists clearly exist"
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured_llm
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(
            question="How many artists are there?",
            sql_query="SELECT COUNT(*) FROM Artist WHERE 1=0",
            sql_result="",
        )
        result = await evaluate_sql_result(state)

        assert "Result is empty but artists clearly exist" in result.sql_error

    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_evaluator_failure_fails_open(self, mock_get_llm_adapter):
        structured_llm = AsyncMock()
        structured_llm.ainvoke.side_effect = RuntimeError("LLM is down")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured_llm
        mock_get_llm_adapter.return_value.get_llm_client.return_value = llm

        state = AgentState(
            question="How many artists are there?",
            sql_query="SELECT COUNT(*) FROM Artist",
            sql_result="275",
        )
        result = await evaluate_sql_result(state)

        assert result.sql_error is None


class TestRouteAfterExecute:
    def test_routes_to_evaluate_sql_result_when_no_error(self):
        state = AgentState(question="q", sql_result="id\n1")
        assert route_after_execute(state) == "evaluate_sql_result"

    def test_routes_to_generate_sql_when_retries_remain(self):
        state = AgentState(question="q", sql_error="boom", retry_count=0)
        assert route_after_execute(state) == "generate_sql"

    def test_routes_to_generate_answer_once_retries_are_exhausted(self):
        state = AgentState(question="q", sql_error="boom", retry_count=MAX_SQL_RETRIES)
        assert route_after_execute(state) == "generate_answer"


class TestRouteAfterEvaluate:
    def test_routes_to_generate_answer_when_no_error(self):
        state = AgentState(question="q", sql_result="id\n1")
        assert route_after_evaluate(state) == "generate_answer"

    def test_routes_to_generate_sql_when_retries_remain(self):
        state = AgentState(
            question="q", sql_error="Self-evaluation rejected", retry_count=0
        )
        assert route_after_evaluate(state) == "generate_sql"

    def test_routes_to_generate_answer_once_retries_are_exhausted(self):
        state = AgentState(
            question="q",
            sql_error="Self-evaluation rejected",
            retry_count=MAX_SQL_RETRIES,
        )
        assert route_after_evaluate(state) == "generate_answer"


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
            ("evaluate_sql_result", evaluate_sql_result),
            ("generate_answer", generate_answer),
        ]
        for name, node in expected_nodes:
            mock_graph.add_node.assert_any_call(name, node)
        assert mock_graph.add_node.call_count == len(expected_nodes)

        mock_graph.add_edge.assert_any_call(START, "generate_sql")
        mock_graph.add_edge.assert_any_call("generate_sql", "execute_sql")
        mock_graph.add_edge.assert_any_call("generate_answer", END)
        mock_graph.add_conditional_edges.assert_any_call(
            "execute_sql",
            route_after_execute,
            {
                "generate_sql": "generate_sql",
                "evaluate_sql_result": "evaluate_sql_result",
                "generate_answer": "generate_answer",
            },
        )
        mock_graph.add_conditional_edges.assert_any_call(
            "evaluate_sql_result",
            route_after_evaluate,
            {"generate_sql": "generate_sql", "generate_answer": "generate_answer"},
        )
        assert mock_graph.add_conditional_edges.call_count == 2

        mock_graph.compile.assert_called_once()
        assert result == mock_compiled_graph
