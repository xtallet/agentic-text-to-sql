from abc import ABC, abstractmethod

from pydantic import BaseModel


class SqlExecutorPort(ABC, BaseModel):
    @abstractmethod
    def get_schema(self) -> str:
        """Return a textual description of the database schema (CREATE TABLE statements)."""

    @abstractmethod
    def execute(self, query: str) -> str:
        """Execute a read-only SELECT query and return the results as a compact string.

        Raises UnsafeSqlError if the query is not a single SELECT statement.
        """
