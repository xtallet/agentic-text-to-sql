from pydantic import BaseModel, Field


class AgentState(BaseModel):
    question: str
    schema_description: str | None = None
    sql_query: str | None = None
    sql_result: str | None = None
    sql_error: str | None = None
    retry_count: int = 0
    failed_attempts: list[str] = Field(default_factory=list)
    answer: str | None = None
