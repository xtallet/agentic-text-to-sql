from pydantic import BaseModel


class AgentState(BaseModel):
    question: str
    schema_description: str | None = None
    sql_query: str | None = None
    sql_result: str | None = None
    sql_error: str | None = None
    answer: str | None = None
