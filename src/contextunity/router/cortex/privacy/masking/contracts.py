"""Runtime-checked contracts for optional masking dependencies.

This module provides Protocol definitions that allow the masking and PII scanning
utilities to interact with third-party libraries (e.g. pandas, presidio-analyzer)
using structural subtyping (duck typing) without requiring hard runtime dependencies
on those libraries.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Collection, Iterable, Iterator
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from contextunity.core.types import JsonDict


@runtime_checkable
class _IlocIndexerLike(Protocol):
    """Protocol mimicking the pandas iloc integer-location based indexer."""

    def __getitem__(self, key: int) -> object:
        """Retrieve a row or value by its integer index.

        Args:
            key: The integer index of the row to retrieve.

        Returns:
            The row data or cell value located at the specified index.
        """
        ...


@runtime_checkable
class PandasSeriesLike(Protocol):
    """Minimal pandas Series interface used by masking and scanning utilities."""

    @property
    def values(self) -> Collection[object]:
        """Get the underlying data values of the series.

        Returns:
            A collection of all values in the series.
        """
        ...

    @property
    def iloc(self) -> _IlocIndexerLike:
        """Get the integer-index locator for accessing elements.

        Returns:
            An indexer supporting integer-based key lookup.
        """
        ...

    def items(self) -> Iterable[tuple[object, object]]:
        """Iterate over (index, value) tuples.

        Returns:
            An iterable of index-value pairs.
        """
        ...

    def apply(self, func: Callable[[object], object]) -> PandasSeriesLike:
        """Apply a function element-wise to the series values.

        Args:
            func: The function to apply to each element.

        Returns:
            A new series containing the transformed values.
        """
        ...

    def __iter__(self) -> Iterator[object]:
        """Return an iterator over the series values.

        Returns:
            An iterator traversing the elements.
        """
        ...


@runtime_checkable
class _AtIndexerLike(Protocol):
    """Minimal pandas .at label-based indexer interface used by masking."""

    def __setitem__(self, key: tuple[object, str], value: object) -> None:
        """Set the value at a specific row label and column name.

        Args:
            key: A tuple containing (row_index, column_name).
            value: The value to assign to the specified cell.
        """
        ...


@runtime_checkable
class PandasFrameLike(Protocol):
    """Minimal pandas DataFrame interface used by masking and scanning utilities."""

    columns: Collection[str]
    at: _AtIndexerLike

    def copy(self) -> PandasFrameLike:
        """Create a deep copy of this dataframe.

        Returns:
            A new DataFrame instance containing cloned data.
        """
        ...

    def drop(self, *, columns: list[str]) -> PandasFrameLike:
        """Return a new dataframe with the specified columns dropped.

        Args:
            columns: A list of column names to remove.

        Returns:
            A new DataFrame instance without the specified columns.
        """
        ...

    def __getitem__(self, key: str) -> PandasSeriesLike:
        """Retrieve a column by name.

        Args:
            key: The name of the column to retrieve.

        Returns:
            The requested column represented as a Series.
        """
        ...

    def __setitem__(self, key: str, value: object) -> None:
        """Assign or update a column's values.

        Args:
            key: The name of the column to set or update.
            value: The data to assign to the column (scalar or sequence).
        """
        ...

    def select_dtypes(self, *, include: list[str]) -> PandasFrameLike:
        """Select a subset of columns matching specified data types.

        Args:
            include: List of dtype strings (e.g. ['object']) to include.

        Returns:
            A new DataFrame containing only matching columns.
        """
        ...

    def __len__(self) -> int:
        """Return the number of rows in the dataframe.

        Returns:
            The row count.
        """
        ...


@runtime_checkable
class PresidioResultLike(Protocol):
    """Minimal interface matching a Presidio AnalyzerEngine recognition result."""

    entity_type: str


@runtime_checkable
class PresidioAnalyzerLike(Protocol):
    """Minimal Presidio AnalyzerEngine interface used by the PII scanner."""

    def analyze(self, text: str, *, language: str) -> list[PresidioResultLike]:
        """Analyze text to identify personally identifiable information (PII) entities.

        Args:
            text: The raw string text to scan.
            language: The ISO language code (e.g. 'en') to use for analysis.

        Returns:
            A list of detected PII entity results.
        """
        ...


MaskingOperation = Literal["mask", "unmask", "scan", "destroy"]
MaskingEntityCounts: TypeAlias = dict[str, int]
MaskingAuditMetadata = JsonDict


@runtime_checkable
class _NullaryFactory(Protocol):
    """Runtime-checkable zero-argument factory used for optional imports."""

    def __call__(self) -> object:
        """Construct and return an object instance."""
        ...


def as_pandas_frame(df: object) -> PandasFrameLike:
    """Validate that an object is a pandas DataFrame and cast it to our protocol.

    Args:
        df: The object to validate and cast.

    Returns:
        The validated DataFrame-like object conforming to the PandasFrameLike protocol.

    Raises:
        TypeError: If the input object is not an instance of pandas.DataFrame.
    """
    pandas_mod = importlib.import_module("pandas")
    dataframe_cls = vars(pandas_mod).get("DataFrame")
    if not isinstance(dataframe_cls, type) or not isinstance(df, dataframe_cls):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    if isinstance(df, PandasFrameLike):
        return df
    raise TypeError("Loaded DataFrame does not satisfy PandasFrameLike contract")


def load_presidio_analyzer() -> PresidioAnalyzerLike:
    """Import ``presidio_analyzer`` at runtime and return a fresh ``AnalyzerEngine`` instance."""
    presidio_mod = importlib.import_module("presidio_analyzer")
    analyzer_cls = vars(presidio_mod).get("AnalyzerEngine")
    if not isinstance(analyzer_cls, _NullaryFactory):
        raise TypeError("presidio_analyzer.AnalyzerEngine is not a class")
    analyzer: object = analyzer_cls()
    if isinstance(analyzer, PresidioAnalyzerLike):
        return analyzer
    raise TypeError("AnalyzerEngine does not satisfy PresidioAnalyzerLike contract")
