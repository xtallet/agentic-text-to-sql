class UnsafeSqlError(Exception):
    """Raised when a generated SQL statement is not a safe, read-only SELECT."""
