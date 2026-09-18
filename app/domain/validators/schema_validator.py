import re

_TABLE_NAME_PATTERN = re.compile(r'CREATE TABLE\s+["\[]?(\w+)["\]]?', re.IGNORECASE)
_REFERENCED_TABLE_PATTERN = re.compile(
    r'\b(?:FROM|JOIN)\s+["\[]?(\w+)["\]]?', re.IGNORECASE
)
_CTE_NAME_PATTERN = re.compile(
    r'(?:WITH|,)\s+["\[]?(\w+)["\]]?\s+AS\s*\(', re.IGNORECASE
)


def validate_sql_schema(sql_query: str, schema_description: str) -> str | None:
    known_tables = {
        name.lower() for name in _TABLE_NAME_PATTERN.findall(schema_description)
    }
    known_tables |= {name.lower() for name in _CTE_NAME_PATTERN.findall(sql_query)}

    referenced_tables = {
        name.lower() for name in _REFERENCED_TABLE_PATTERN.findall(sql_query)
    }

    unknown = referenced_tables - known_tables
    if unknown:
        return f"Unknown table(s) referenced: {', '.join(sorted(unknown))}"
    return None
