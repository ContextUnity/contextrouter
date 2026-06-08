class DuckDBPyConnection:
    def execute(
        self,
        query: object,
        parameters: object | None = None,
    ) -> DuckDBPyConnection: ...
    def fetchall(self) -> list[tuple[object, ...]]: ...
    def fetchone(self) -> tuple[object, ...] | None: ...

def connect(
    database: str = ...,
    *,
    read_only: bool = False,
) -> DuckDBPyConnection: ...

__all__ = ["DuckDBPyConnection", "connect"]
