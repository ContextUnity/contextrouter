"""Tests for ingestion batch processing utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from contextrouter.core.config import Config
from contextrouter.modules.ingestion.rag import (
    BatchResult,
    batch_transform,
    batch_validate,
    chunked,
    filter_by_indices,
)
from contextrouter.modules.ingestion.rag.core.batch import _parse_validation_response


@pytest.fixture()
def core_cfg() -> Config:
    # Minimal config accepted by VertexLLM (used by ingestion llm utilities).
    return Config.model_validate({"vertex": {"project_id": "test", "location": "us-central1"}})


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_success_indices_returns_set(self) -> None:
        result = BatchResult(successes=[(0, "a"), (2, "b"), (5, "c")], failures=[1, 3])
        assert result.success_indices == {0, 2, 5}

    def test_success_indices_empty(self) -> None:
        result = BatchResult[str](successes=[], failures=[0, 1])
        assert result.success_indices == set()

    def test_get_returns_value_for_existing_index(self) -> None:
        result = BatchResult(successes=[(0, "first"), (2, "third")], failures=[1])
        assert result.get(0) == "first"
        assert result.get(2) == "third"

    def test_get_returns_default_for_missing_index(self) -> None:
        result = BatchResult(successes=[(0, "first")], failures=[1])
        assert result.get(1) is None
        assert result.get(1, "default") == "default"
        assert result.get(99, "not found") == "not found"


class TestParseValidationResponse:
    """Tests for _parse_validation_response helper."""

    def test_parses_tab_separated_response(self) -> None:
        valid: set[int] = set()
        batch = [(0, "item0"), (1, "item1"), (2, "item2")]
        response = "0\tVALUABLE\n1\tNOT_VALUABLE\n2\tVALUABLE"

        # Use exact match to avoid "VALUABLE" matching "NOT_VALUABLE"
        _parse_validation_response(
            response, batch, valid, lambda s: s.strip().upper() == "VALUABLE"
        )

        assert valid == {0, 2}

    def test_parses_literal_tab_marker(self) -> None:
        """LLM sometimes outputs <TAB> literally instead of actual tabs."""
        valid: set[int] = set()
        batch = [(0, "item0"), (1, "item1")]
        response = "0<TAB>VALUABLE\n1<TAB>NOT_VALUABLE"

        _parse_validation_response(
            response, batch, valid, lambda s: s.strip().upper() == "VALUABLE"
        )

        assert valid == {0}

    def test_ignores_empty_lines(self) -> None:
        valid: set[int] = set()
        batch = [(0, "item0"), (1, "item1")]
        response = "\n0\tVALUABLE\n\n\n1\tVALUABLE\n"

        _parse_validation_response(response, batch, valid, lambda s: "VALUABLE" in s.upper())

        assert valid == {0, 1}

    def test_ignores_malformed_lines(self) -> None:
        valid: set[int] = set()
        batch = [(0, "item0"), (1, "item1")]
        response = "not-an-index\tVALUABLE\n0\tVALUABLE\nincomplete"

        _parse_validation_response(response, batch, valid, lambda s: "VALUABLE" in s.upper())

        assert valid == {0}

    def test_custom_parse_decision(self) -> None:
        valid: set[int] = set()
        batch = [(0, "item0"), (1, "item1")]
        response = "0\tKEEP\n1\tDROP"

        _parse_validation_response(response, batch, valid, lambda s: s.strip().upper() == "KEEP")

        assert valid == {0}


class TestBatchValidate:
    """Tests for batch_validate function."""

    def test_returns_empty_set_for_empty_input(self, core_cfg: Config) -> None:
        result = batch_validate(
            [],
            core_cfg=core_cfg,
            prompt_builder=lambda x: "unused",
            label="test",
        )
        assert result == set()

    @patch("contextrouter.modules.ingestion.rag.core.batch.llm_generate")
    def test_validates_items_in_batches(self, mock_llm: MagicMock, core_cfg: Config) -> None:
        mock_llm.return_value = "0\tVALUABLE\n1\tNOT_VALUABLE\n2\tVALUABLE"
        items = ["item0", "item1", "item2"]

        result = batch_validate(
            items,
            core_cfg=core_cfg,
            prompt_builder=lambda batch: f"Validate: {batch}",
            batch_size=10,
            label="test",
            parse_decision=lambda s: s.strip().upper() == "VALUABLE",
        )

        assert result == {0, 2}
        mock_llm.assert_called_once()

    @patch("contextrouter.modules.ingestion.rag.core.batch.llm_generate")
    def test_splits_into_batches(self, mock_llm: MagicMock, core_cfg: Config) -> None:
        # First batch: all valid, second batch: one valid
        mock_llm.side_effect = [
            "0\tVALUABLE\n1\tVALUABLE",
            "2\tNOT_VALUABLE\n3\tVALUABLE",
        ]
        items = ["a", "b", "c", "d"]

        result = batch_validate(
            items,
            core_cfg=core_cfg,
            prompt_builder=lambda batch: str(batch),
            batch_size=2,
            label="test",
            parse_decision=lambda s: s.strip().upper() == "VALUABLE",
        )

        assert result == {0, 1, 3}
        assert mock_llm.call_count == 2

    @patch("contextrouter.modules.ingestion.rag.core.batch.llm_generate")
    def test_on_error_keep_preserves_all(self, mock_llm: MagicMock, core_cfg: Config) -> None:
        mock_llm.side_effect = Exception("LLM error")
        items = ["item0", "item1"]

        result = batch_validate(
            items,
            core_cfg=core_cfg,
            prompt_builder=lambda batch: str(batch),
            on_error="keep",
            label="test",
        )

        assert result == {0, 1}

    @patch("contextrouter.modules.ingestion.rag.core.batch.llm_generate")
    def test_on_error_drop_removes_all(self, mock_llm: MagicMock, core_cfg: Config) -> None:
        mock_llm.side_effect = Exception("LLM error")
        items = ["item0", "item1"]

        result = batch_validate(
            items,
            core_cfg=core_cfg,
            prompt_builder=lambda batch: str(batch),
            on_error="drop",
            label="test",
        )

        assert result == set()


class TestBatchTransform:
    """Tests for batch_transform function."""

    def test_returns_empty_dict_for_empty_input(self, core_cfg: Config) -> None:
        result = batch_transform(
            [],
            core_cfg=core_cfg,
            prompt_builder=lambda x: "unused",
            result_parser=lambda line, idx: line,
            label="test",
        )
        assert result == {}

    @patch("contextrouter.modules.ingestion.rag.core.batch.llm_generate")
    def test_transforms_items_and_parses_results(
        self, mock_llm: MagicMock, core_cfg: Config
    ) -> None:
        mock_llm.return_value = "0\tsummary_0\n1\tsummary_1"
        items = ["content0", "content1"]

        def parser(line: str, idx: int) -> str | None:
            parts = line.split("\t")
            return parts[1] if len(parts) > 1 else None

        result = batch_transform(
            items,
            core_cfg=core_cfg,
            prompt_builder=lambda batch: str(batch),
            result_parser=parser,
            label="test",
        )

        assert result == {0: "summary_0", 1: "summary_1"}

    @patch("contextrouter.modules.ingestion.rag.core.batch.llm_generate")
    def test_skips_items_when_parser_returns_none(
        self, mock_llm: MagicMock, core_cfg: Config
    ) -> None:
        mock_llm.return_value = "0\tgood\n1\t\n2\tgood"  # idx 1 is empty
        items = ["a", "b", "c"]

        def parser(line: str, idx: int) -> str | None:
            parts = line.split("\t")
            return parts[1] if len(parts) > 1 and parts[1].strip() else None

        result = batch_transform(
            items,
            core_cfg=core_cfg,
            prompt_builder=lambda batch: str(batch),
            result_parser=parser,
            label="test",
        )

        assert result == {0: "good", 2: "good"}


class TestChunked:
    """Tests for chunked utility."""

    def test_chunks_evenly_divisible(self) -> None:
        result = list(chunked([1, 2, 3, 4, 5, 6], size=2))
        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_chunks_with_remainder(self) -> None:
        result = list(chunked([1, 2, 3, 4, 5], size=2))
        assert result == [[1, 2], [3, 4], [5]]

    def test_empty_iterable(self) -> None:
        result = list(chunked([], size=10))
        assert result == []

    def test_single_item(self) -> None:
        result = list(chunked([1], size=10))
        assert result == [[1]]

    def test_chunk_size_larger_than_input(self) -> None:
        result = list(chunked([1, 2], size=10))
        assert result == [[1, 2]]

    def test_works_with_generator(self) -> None:
        gen = (x for x in range(5))
        result = list(chunked(gen, size=2))
        assert result == [[0, 1], [2, 3], [4]]


class TestFilterByIndices:
    """Tests for filter_by_indices utility."""

    def test_filters_by_valid_indices(self) -> None:
        items = ["a", "b", "c", "d", "e"]
        valid = {0, 2, 4}
        result = filter_by_indices(items, valid)
        assert result == ["a", "c", "e"]

    def test_empty_valid_indices(self) -> None:
        items = ["a", "b", "c"]
        result = filter_by_indices(items, set())
        assert result == []

    def test_all_valid_indices(self) -> None:
        items = ["a", "b", "c"]
        result = filter_by_indices(items, {0, 1, 2})
        assert result == ["a", "b", "c"]

    def test_indices_beyond_range_are_ignored(self) -> None:
        items = ["a", "b"]
        result = filter_by_indices(items, {0, 1, 99})
        assert result == ["a", "b"]

    def test_maintains_original_order(self) -> None:
        items = ["a", "b", "c", "d"]
        valid = {3, 1}  # Unordered set
        result = filter_by_indices(items, valid)
        assert result == ["b", "d"]  # Original order preserved
