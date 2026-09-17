import pytest

from app.domain.exceptions.sql_exceptions import UnsafeSqlError
from app.infrastructure.adapters.sqlite_adapter import SqliteAdapter

DB_PATH = "data/chinook.db"


@pytest.fixture
def adapter() -> SqliteAdapter:
    return SqliteAdapter(db_path=DB_PATH)


class TestSqliteAdapter:
    def test_get_schema_includes_known_tables(self, adapter):
        schema = adapter.get_schema()

        assert "CREATE TABLE" in schema
        assert "Artist" in schema
        assert "Invoice" in schema

    def test_execute_select_returns_rows(self, adapter):
        result = adapter.execute("SELECT Name FROM Artist WHERE ArtistId = 1")

        assert "Name" in result
        assert "AC/DC" in result

    @pytest.mark.parametrize(
        "query",
        [
            "DROP TABLE Artist",
            "DELETE FROM Artist",
            "UPDATE Artist SET Name = 'x'",
            "SELECT 1; DROP TABLE Artist",
            "PRAGMA table_info(Artist)",
        ],
    )
    def test_execute_rejects_non_select_statements(self, adapter, query):
        with pytest.raises(UnsafeSqlError):
            adapter.execute(query)
