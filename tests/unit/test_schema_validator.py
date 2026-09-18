from app.domain.validators.schema_validator import validate_sql_schema

SCHEMA = """
CREATE TABLE [Artist]
(
    [ArtistId] INTEGER NOT NULL,
    [Name] NVARCHAR(120)
)

CREATE TABLE [Album]
(
    [AlbumId] INTEGER NOT NULL,
    [Title] NVARCHAR(160) NOT NULL,
    [ArtistId] INTEGER NOT NULL
)
"""


def test_valid_single_table_query_passes():
    assert validate_sql_schema("SELECT * FROM Artist", SCHEMA) is None


def test_valid_join_query_passes():
    query = "SELECT * FROM Album a JOIN Artist ar ON a.ArtistId = ar.ArtistId"
    assert validate_sql_schema(query, SCHEMA) is None


def test_unknown_table_is_rejected():
    result = validate_sql_schema("SELECT * FROM NotExistingTable", SCHEMA)

    assert result is not None
    assert "notexistingtable" in result.lower()


def test_cte_name_is_not_flagged_as_unknown_table():
    query = "WITH RankedArtists AS (SELECT * FROM Artist) SELECT * FROM RankedArtists"
    assert validate_sql_schema(query, SCHEMA) is None
