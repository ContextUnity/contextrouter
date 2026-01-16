"""Question analysis component for QA plugin."""

from __future__ import annotations

import logging
from typing import Any

from contextrouter.core import Config

from ...utils.llm import llm_generate

logger = logging.getLogger(__name__)


class QuestionAnalyzer:
    """Handles question extraction, validation, and analysis."""

    def __init__(self, core_cfg: Config) -> None:
        self.core_cfg = core_cfg

    def _validate_question_with_llm(self, question_text: str) -> bool:
        """Use LLM to validate if text is a proper question."""
        try:
            # Import here to avoid circular imports
            from ..core.prompts import qa_validate_question_prompt

            prompt = qa_validate_question_prompt()
            response = llm_generate(
                core_cfg=self.core_cfg,
                prompt=prompt,
                content=question_text,
                model=self.core_cfg.models.ingestion.preprocess.model,
                max_tokens=100,
                temperature=0.0,
            )

            # Simple yes/no check
            response_lower = response.lower().strip()
            return "yes" in response_lower or "true" in response_lower

        except Exception as e:
            logger.debug("LLM question validation failed: %s", e)
            # Fallback to simple heuristics
            return self._is_question_like(question_text)

    def _is_question_like(self, text: str) -> bool:
        """Simple heuristic to check if text looks like a question."""
        text = text.strip()
        if not text:
            return False

        # Check for question marks
        if "?" in text:
            return True

        # Check for question words
        question_words = [
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "can",
            "could",
            "would",
            "should",
            "do",
            "does",
            "did",
            "is",
            "are",
            "was",
            "were",
        ]
        first_word = text.lower().split()[0] if text else ""

        return first_word in question_words

    def _validate_answer_with_llm(self, question_text: str, answer_text: str) -> bool:
        """Use LLM to validate if answer is valuable for the question."""
        try:
            # Import here to avoid circular imports
            from ..core.prompts import qa_validate_answer_prompt

            prompt = qa_validate_answer_prompt()
            content = f"Question: {question_text}\nAnswer: {answer_text}"

            response = llm_generate(
                core_cfg=self.core_cfg,
                prompt=prompt,
                content=content,
                model=self.core_cfg.models.ingestion.preprocess.model,
                max_tokens=100,
                temperature=0.0,
            )

            # Simple yes/no check
            response_lower = response.lower().strip()
            return "yes" in response_lower or "true" in response_lower

        except Exception as e:
            logger.debug("LLM answer validation failed: %s (keeping answer)", e)
            # Default to keeping the answer
            return True

    def _analyze_segments_batch(self, chunks: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
        """Analyze conversation segments in batch using LLM."""
        try:
            # Import here to avoid circular imports
            from ..core.prompts import qa_batch_analysis_prompt

            # Prepare content for batch analysis
            content_parts = []
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict) and "text" in chunk:
                    content_parts.append(f"[{i}] {chunk['text']}")

            content = "\n".join(content_parts)

            prompt = qa_batch_analysis_prompt()
            response = llm_generate(
                core_cfg=self.core_cfg,
                prompt=prompt,
                content=content,
                model=self.core_cfg.models.ingestion.preprocess.model,
                max_tokens=2048,
                temperature=0.0,
            )

            return self._parse_batch_analysis(response, chunks)

        except Exception as e:
            logger.warning("LLM batch analysis failed: %s, using fallback", e)
            return self._fallback_analysis(chunks)

    def _parse_batch_analysis(
        self, response: str, chunks: list[dict[str, Any]]
    ) -> dict[int, dict[str, str]]:
        """Parse batch analysis response."""
        results: dict[int, dict[str, str]] = {}

        # Simple parsing - can be enhanced
        for i, chunk in enumerate(chunks):
            results[i] = {
                "question": chunk.get("text", "")[:100],  # Placeholder
                "answer": chunk.get("text", "")[100:],  # Placeholder
            }

        return results

    def _fallback_analysis(self, chunks: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
        """Fallback analysis when LLM fails."""
        results: dict[int, dict[str, str]] = {}

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            # Simple split: first sentence as question, rest as answer
            sentences = text.split(".", 1)
            if len(sentences) >= 2:
                question = sentences[0].strip() + "?"
                answer = sentences[1].strip()
            else:
                question = text[:50] + "..."
                answer = text[50:]

            results[i] = {
                "question": question,
                "answer": answer,
            }

        return results
