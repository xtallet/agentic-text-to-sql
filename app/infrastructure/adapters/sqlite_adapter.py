import sqlite3
from pathlib import Path

from app.domain.exceptions.sql_exceptions import UnsafeSqlError
from app.domain.ports.sql_executor_port import SqlExecutorPort

MAX_ROWS = 200

PII_COLUMNS = {"email", "phone", "fax", "address", "birthdate", "postalcode"}
REDACTED = "[REDACTED]"


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return str(round(value, 2))
    return str(value)


class SqliteAdapter(SqlExecutorPort):
    db_path: str

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{Path(self.db_path).resolve()}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def get_schema(self) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL"
            ).fetchall()
        return "\n\n".join(row[0] for row in rows)

    def execute(self, query: str) -> str:
        statements = [
            s.strip() for s in query.strip().rstrip(";").split(";") if s.strip()
        ]
        if len(statements) != 1 or not statements[0].lower().startswith("select"):
            raise UnsafeSqlError("Only a single read-only SELECT statement is allowed.")

        with self._connect() as conn:
            cursor = conn.execute(statements[0])
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS)

        if not rows:
            return "(no rows returned)"

        pii_indexes = {
            i for i, name in enumerate(columns) if name.lower() in PII_COLUMNS
        }

        lines = [", ".join(columns)]
        for row in rows:
            values = [
                REDACTED
                if i in pii_indexes and value is not None
                else _format_value(value)
                for i, value in enumerate(row)
            ]
            lines.append(", ".join(values))
        return "\n".join(lines)
