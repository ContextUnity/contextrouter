"""Tests for ingestion prompt templates."""

from __future__ import annotations

from contextrouter.modules.ingestion.rag import (
    qa_rephrase_question_prompt,
    qa_validate_answer_prompt,
    qa_validate_question_prompt,
    video_validate_segment_prompt,
)


class TestQaValidateQuestionPrompt:
    """Tests for qa_validate_question_prompt."""

    def test_includes_raw_text_in_prompt(self) -> None:
        prompt = qa_validate_question_prompt(
            raw_text="What is the mastermind principle?",
            answer_context="The mastermind principle involves combining minds...",
        )

        assert "What is the mastermind principle?" in prompt
        assert "mastermind principle involves" in prompt

    def test_truncates_long_answer_context(self) -> None:
        long_answer = "A" * 1000
        prompt = qa_validate_question_prompt(
            raw_text="Question?",
            answer_context=long_answer,
        )

        # Answer context should be truncated to 500 chars
        assert long_answer not in prompt
        assert "A" * 500 in prompt

    def test_includes_rejection_criteria(self) -> None:
        prompt = qa_validate_question_prompt(
            raw_text="test",
            answer_context="test",
        )

        # Should include key rejection criteria
        assert "NOT_A_QUESTION" in prompt
        assert "greeting" in prompt.lower() or "pleasantry" in prompt.lower()
        assert "promotional" in prompt.lower()

    def test_includes_format_instructions(self) -> None:
        prompt = qa_validate_question_prompt(
            raw_text="test",
            answer_context="test",
        )

        # Should include format instructions
        assert "NOT_A_QUESTION" in prompt
        assert "max" in prompt.lower() or "character" in prompt.lower()


class TestQaValidateAnswerPrompt:
    """Tests for qa_validate_answer_prompt."""

    def test_includes_answer_text(self) -> None:
        prompt = qa_validate_answer_prompt(
            answer_text="The three principles are persistence, clarity, and action.",
            topic="Success Principles",
        )

        assert "three principles are persistence" in prompt

    def test_includes_topic(self) -> None:
        prompt = qa_validate_answer_prompt(
            answer_text="content",
            topic="Mastermind Principle",
        )

        assert "Mastermind Principle" in prompt

    def test_includes_valuable_not_valuable_criteria(self) -> None:
        prompt = qa_validate_answer_prompt(answer_text="test", topic="test")

        assert "VALUABLE" in prompt
        assert "NOT_VALUABLE" in prompt
        assert "administrative" in prompt.lower() or "logistical" in prompt.lower()
        assert "promotional" in prompt.lower()

    def test_output_format_is_clear(self) -> None:
        prompt = qa_validate_answer_prompt(answer_text="test", topic="test")

        # Should specify expected output format
        assert "VALUABLE" in prompt and "NOT_VALUABLE" in prompt


class TestVideoValidateSegmentPrompt:
    """Tests for video_validate_segment_prompt."""

    def test_includes_segment_text_and_title(self) -> None:
        prompt = video_validate_segment_prompt(
            segment_text="This is about persistence...",
            video_title="Success Principles Episode 1",
        )

        assert "This is about persistence" in prompt
        assert "Success Principles Episode 1" in prompt

    def test_truncates_long_segment_text(self) -> None:
        long_text = "X" * 2000
        prompt = video_validate_segment_prompt(
            segment_text=long_text,
            video_title="Test",
        )

        # Should truncate to 1500 chars
        assert "X" * 1500 in prompt
        assert long_text not in prompt

    def test_includes_rejection_categories(self) -> None:
        prompt = video_validate_segment_prompt(
            segment_text="test",
            video_title="test",
        )

        # Key rejection categories
        assert "TECHNICAL" in prompt or "technical" in prompt
        assert "HOUSEKEEPING" in prompt or "housekeeping" in prompt
        assert "PROMOTIONAL" in prompt or "promotional" in prompt


class TestQaRephraseQuestionPrompt:
    """Tests for qa_rephrase_question_prompt."""

    def test_includes_question_and_answer(self) -> None:
        prompt = qa_rephrase_question_prompt(
            question="What about that?",
            answer="The mastermind principle works by combining minds.",
        )

        assert "What about that?" in prompt
        assert "mastermind principle" in prompt

    def test_truncates_long_answer(self) -> None:
        long_answer = "B" * 1000
        prompt = qa_rephrase_question_prompt(
            question="Question?",
            answer=long_answer,
        )

        # Should truncate to ~500 chars
        assert long_answer not in prompt

    def test_includes_examples(self) -> None:
        prompt = qa_rephrase_question_prompt(
            question="test",
            answer="test",
        )

        assert "Examples:" in prompt or "Example:" in prompt

    def test_includes_rules_for_rephrasing(self) -> None:
        prompt = qa_rephrase_question_prompt(
            question="test",
            answer="test",
        )

        # Should include rules about clarity
        assert "clear" in prompt.lower()
        assert "self-contained" in prompt.lower()


class TestPromptConsistency:
    """Cross-cutting tests for prompt consistency."""

    def test_validation_prompts_have_consistent_output_format(self) -> None:
        """All validation prompts should have clear output format."""
        q_prompt = qa_validate_question_prompt(raw_text="x", answer_context="y")
        a_prompt = qa_validate_answer_prompt(answer_text="x", topic="y")
        v_prompt = video_validate_segment_prompt(segment_text="x", video_title="y")

        # Answer and video validation should specify binary output
        for prompt in [a_prompt, v_prompt]:
            assert "VALUABLE" in prompt
            assert "NOT_VALUABLE" in prompt

        # Question validation has different output (question text or NOT_A_QUESTION)
        assert "NOT_A_QUESTION" in q_prompt

    def test_prompts_are_not_empty(self) -> None:
        """Sanity check that prompts return non-empty strings."""
        prompts = [
            qa_validate_question_prompt(raw_text="x", answer_context="y"),
            qa_validate_answer_prompt(answer_text="x", topic="y"),
            video_validate_segment_prompt(segment_text="x", video_title="y"),
            qa_rephrase_question_prompt(question="x", answer="y"),
        ]

        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Reasonable minimum length
