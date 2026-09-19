from app.infrastructure.services.langsmith import build_langsmith_config


def test_includes_question_in_metadata():
    config = build_langsmith_config("How many tracks are there?")

    assert config["metadata"]["question"] == "How many tracks are there?"


def test_includes_llm_model_in_metadata():
    config = build_langsmith_config("How many tracks are there?")

    assert "llm_model" in config["metadata"]


def test_includes_project_tags():
    config = build_langsmith_config("How many tracks are there?")

    assert "agentic-text-to-sql" in config["tags"]
    assert "cli" in config["tags"]


def test_run_name_includes_the_question():
    config = build_langsmith_config("How many tracks are there?")

    assert "How many tracks are there?" in config["run_name"]


def test_run_name_truncates_long_questions():
    long_question = "a" * 200

    config = build_langsmith_config(long_question)

    assert len(config["run_name"]) < len(long_question)
