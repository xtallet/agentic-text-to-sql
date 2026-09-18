import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langgraph.types import Command

from app.graph import (
    MAX_SQL_RETRIES,
    AmbiguityCheck,
    AnswerEvaluation,
    SqlEvaluation,
    SqlQuery,
    compile_graph,
)


class ScriptedLlmClient:
    """A fake LangChain chat model that returns canned responses in order.

    Mirrors the real client's interface just enough for our nodes: nodes either call
    `with_structured_output(Model).ainvoke(messages)` or `ainvoke(messages)` directly.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if not self._script:
            raise AssertionError("ScriptedLlmClient ran out of scripted responses")
        return self._script.pop(0)


class ScriptedLlmAdapter:
    def __init__(self, script):
        self.client = ScriptedLlmClient(script)

    def get_llm_client(self):
        return self.client


def _config():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


class TestHappyPath:
    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_simple_question_is_answered_without_retries(
        self, mock_get_llm_adapter
    ):
        script = [
            AmbiguityCheck(is_ambiguous=False),
            SqlQuery(query="SELECT COUNT(*) AS TrackCount FROM Track"),
            SqlEvaluation(is_valid=True, reason="Matches the question"),
            SimpleNamespace(content="There are 3,503 tracks in the database."),
            AnswerEvaluation(is_valid=True, reason="Faithful to the result"),
        ]
        mock_get_llm_adapter.return_value = ScriptedLlmAdapter(script)

        graph = await compile_graph()
        result = await graph.ainvoke(
            {"question": "How many tracks are there?"}, _config()
        )

        assert result["answer"] == "There are 3,503 tracks in the database."
        assert "3503" in result["sql_result"]
        assert result["retry_count"] == 0
        assert result["sql_error"] is None


class TestSqlRetryRecovery:
    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_recovers_after_one_bad_sql_attempt(self, mock_get_llm_adapter):
        script = [
            AmbiguityCheck(is_ambiguous=False),
            SqlQuery(query="SELECT * FROM NotExistingTable"),
            SqlQuery(query="SELECT COUNT(*) AS TrackCount FROM Track"),
            SqlEvaluation(is_valid=True, reason="Matches the question"),
            SimpleNamespace(content="There are 3,503 tracks in the database."),
            AnswerEvaluation(is_valid=True, reason="Faithful to the result"),
        ]
        mock_get_llm_adapter.return_value = ScriptedLlmAdapter(script)

        graph = await compile_graph()
        result = await graph.ainvoke(
            {"question": "How many tracks are there?"}, _config()
        )

        assert result["answer"] == "There are 3,503 tracks in the database."
        assert result["retry_count"] == 1
        assert len(result["failed_attempts"]) == 1
        assert "notexistingtable" in result["failed_attempts"][0].lower()


class TestSqlRetryExhaustion:
    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_gives_up_gracefully_after_max_retries(self, mock_get_llm_adapter):
        bad_query = SqlQuery(query="SELECT * FROM NotExistingTable")
        script = [
            AmbiguityCheck(is_ambiguous=False),
            bad_query,
            bad_query,
            bad_query,
            SimpleNamespace(
                content="Sorry, I could not find a valid table to answer that."
            ),
            AnswerEvaluation(is_valid=True, reason="Faithfully reports the failure"),
        ]
        mock_get_llm_adapter.return_value = ScriptedLlmAdapter(script)

        graph = await compile_graph()
        result = await graph.ainvoke(
            {"question": "How many tracks are there?"}, _config()
        )

        assert result["retry_count"] == MAX_SQL_RETRIES
        assert result["sql_error"] is not None
        assert (
            result["answer"] == "Sorry, I could not find a valid table to answer that."
        )


class TestSelfEvaluationRejectionRecovery:
    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_recovers_after_evaluator_rejects_first_result(
        self, mock_get_llm_adapter
    ):
        valid_query = SqlQuery(query="SELECT COUNT(*) AS TrackCount FROM Track")
        script = [
            AmbiguityCheck(is_ambiguous=False),
            valid_query,
            SqlEvaluation(is_valid=False, reason="Testing rejection"),
            valid_query,
            SqlEvaluation(is_valid=True, reason="Matches the question"),
            SimpleNamespace(content="There are 3,503 tracks in the database."),
            AnswerEvaluation(is_valid=True, reason="Faithful to the result"),
        ]
        mock_get_llm_adapter.return_value = ScriptedLlmAdapter(script)

        graph = await compile_graph()
        result = await graph.ainvoke(
            {"question": "How many tracks are there?"}, _config()
        )

        assert result["answer"] == "There are 3,503 tracks in the database."
        assert result["retry_count"] == 1
        assert result["sql_error"] is None


class TestHumanInTheLoop:
    @pytest.mark.asyncio
    @patch("app.graph.get_llm_adapter")
    async def test_ambiguous_question_pauses_and_resumes_with_clarification(
        self, mock_get_llm_adapter
    ):
        script = [
            AmbiguityCheck(
                is_ambiguous=True,
                clarifying_question="Which year's Q3 do you mean?",
            ),
            # check_ambiguity re-runs in full on resume, so a second canned
            # classification is needed before interrupt() returns the resume value.
            AmbiguityCheck(
                is_ambiguous=True,
                clarifying_question="Which year's Q3 do you mean?",
            ),
            SqlQuery(query="SELECT COUNT(*) AS TrackCount FROM Track"),
            SqlEvaluation(is_valid=True, reason="Matches the question"),
            SimpleNamespace(content="There are 3,503 tracks in the database."),
            AnswerEvaluation(is_valid=True, reason="Faithful to the result"),
        ]
        adapter = ScriptedLlmAdapter(script)
        mock_get_llm_adapter.return_value = adapter

        graph = await compile_graph()
        config = _config()
        result = await graph.ainvoke(
            {"question": "Which employee sold the most invoices in Q3?"}, config
        )

        assert "__interrupt__" in result
        assert result["__interrupt__"][0].value["question"] == (
            "Which year's Q3 do you mean?"
        )

        result = await graph.ainvoke(Command(resume="2012"), config)

        assert result["clarification"] == "2012"
        assert result["answer"] == "There are 3,503 tracks in the database."

        sql_generation_prompt = adapter.client.calls[2][1].content
        assert "User clarification: 2012" in sql_generation_prompt
