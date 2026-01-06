"""Centralized prompt templates for ingestion-time LLM calls.

This module provides reusable prompt templates for various LLM operations
during the ingestion pipeline:

- **Content Validation**: Filter out non-valuable content before indexing
  - `qa_validate_question_prompt`: Filter non-questions (filler, promotional)
  - `qa_validate_answer_prompt`: Filter meaningless answers
  - `video_validate_segment_prompt`: Filter non-educational video segments

- **Content Enhancement**: Improve content quality for search
  - `qa_rephrase_question_prompt`: Clarify unclear questions
  - `web_summary_prompt`: Generate page summaries

- **Analysis**: Extract structure and metadata
  - `qa_speaker_detection_prompt`: Identify speakers in transcripts
  - `qa_batch_analysis_prompt`: Extract topics from QA content
  - `book_batch_analysis_prompt`: Extract topics from book chapters

All prompts follow consistent patterns:
- Clear input/output format specification
- Explicit accept/reject criteria
- Examples where helpful for LLM guidance
"""

from __future__ import annotations


def qa_speaker_detection_prompt(*, transcript: str) -> str:
    return f"""Analyze this conversational transcript and identify all speaker segments.

TRANSCRIPT:
{transcript}

Extract all speaker segments. For each segment:
1. Identify the speaker's full canonical name
2. Extract the exact text spoken (preserve original, don't summarize)

Return as TSV (tab-separated), one segment per line:
speaker_name<TAB>exact_text_spoken

Rules:
- Use REAL tab characters (\\t), not the text "<TAB>"
- Use full canonical names consistently
- Preserve EXACT text content (no summarizing)
- Skip segments shorter than 10 characters
- Maintain chronological order
- Use "Unknown" if speaker unclear

Example output:
Speaker A\tWelcome everyone to today's session on desire...
Speaker B\tThank you. Let's dive into the principles...
Speaker A\tExactly. The first principle is..."""


def qa_batch_analysis_prompt(
    *, items_text: str, taxonomy_categories: list[str] | None = None
) -> str:
    category_instruction = ""
    if taxonomy_categories:
        cat_list = ", ".join(taxonomy_categories)
        category_instruction = f"\n- category: Best-matching from: [{cat_list}]"

    return f"""Analyze these conversational transcript segments.

SEGMENTS:
{items_text}

For each chunk (CHUNK 0, CHUNK 1, ...), extract:
- topic: Concise 5-8 word summary
- primary_speaker: Speaker who dominates the segment{category_instruction}

Return as TSV, one line per chunk:
index<TAB>topic<TAB>primary_speaker<TAB>category

Rules:
- Use REAL tab characters (\\t), not "<TAB>"
- category can be empty if no taxonomy provided
- Maintain chunk order (0, 1, 2, ...)

Example:
0\tBuilding habits for success\tSpeaker A\tpersonal_development
1\tOvercoming fear and doubt\tSpeaker B\tmindset"""


def web_summary_prompt(*, content: str, categories: list[str] | None = None) -> str:
    taxonomy_context = ""
    if categories:
        taxonomy_context = f"\nCategories to focus on: {', '.join(categories)}"
    return f"""Summarize this web content in 1-2 sentences.
Focus on the main topic and key insights.{taxonomy_context}

CONTENT:
{content}

Return ONLY the summary text (no JSON, no quotes)."""


def qa_validate_question_prompt(*, raw_text: str, answer_context: str) -> str:
    """Prompt to validate and extract meaningful questions from conversation text.

    Filters out:
    - Non-questions (statements, declarations)
    - Promotional/administrative content (pricing, access, signups)
    - Conversational filler (greetings, pleasantries, acknowledgments)
    - Meta-commentary about the session itself
    - Auto-generated summaries / topic headers
    - Mid-sentence fragments

    Returns either a clean question or "NOT_A_QUESTION".
    """
    return f"""Analyze this text and determine if it contains a GENUINE QUESTION worth indexing.

TEXT TO ANALYZE:
{raw_text}

ANSWER CONTEXT (for understanding the topic):
{answer_context[:500]}

TASK: Determine if the text contains a genuine, substantive question that someone might search for.

REJECT (return "NOT_A_QUESTION") if the text is:

1. NOT A QUESTION - statements, declarations, opinions without inquiry
   - "Yeah, so it's a free program" → NOT_A_QUESTION
   - "I'm excited about this book" → NOT_A_QUESTION
   - "That's a great point" → NOT_A_QUESTION
   - "and it's always there. But if we're not letting it in..." → NOT_A_QUESTION (fragment)

2. TOPIC DESCRIPTIONS (not actual questions):
   - "[Speaker] discussing [topic]..." → NOT_A_QUESTION
   - Descriptive headers that aren't real inquiries

3. MID-THOUGHT FRAGMENTS:
   - Text starting with lowercase "and", "but", "or", "so" that's clearly mid-sentence
   - Incomplete sentences or thought fragments
   - Text that doesn't stand alone as a coherent question

4. VAGUE TAG QUESTIONS or rhetorical filler:
   - "isn't there?", "don't you think?", "you know?" (as standalone)
   - "Yeah, there's energy in that isn't there?" → NOT_A_QUESTION
   - Questions that are just conversational prompts, not real inquiries

5. PROMOTIONAL/ADMINISTRATIVE content:
   - Pricing, access periods, signups ("free program", "72 hours", "join for free")
   - Program logistics ("watch the videos", "go dark", "companion app")
   - Marketing speak ("excited to have you", "welcome to the community")
   - Book recommendations/purchases ("pick this book up on Amazon")

6. CONVERSATIONAL FILLER:
   - Greetings: "Hello", "Welcome", "Hey everyone", "Good morning"
   - Pleasantries: "Thank you", "Great", "Absolutely", "Right"
   - Acknowledgments: "Yeah", "Okay", "I see", "Got it"
   - Transitional: "So", "Now", "Well", "Alright"

7. META-COMMENTARY about the session:
   - "We're going to get started", "Are we halfway?"
   - "Let me share my screen", "Can everyone hear me?"

ACCEPT (extract the question) if it's:
- A genuine, complete inquiry about concepts, principles, methods, or ideas
- Something a learner might actually search for
- Related to the educational/instructional content
- Stands alone as a coherent, meaningful question

OUTPUT FORMAT:
- If valid question found: Return ONLY the question text (cleaned, max 200 chars)
- If not a valid question: Return exactly "NOT_A_QUESTION"

EXAMPLES:
Text: "Yeah, so it's a free program. As you know you joined for free and you get access for 72 hours."
Output: NOT_A_QUESTION

Text: "What are the three principles mentioned in the book?"
Output: What are the three principles mentioned in the book?

Text: "and it's always there. But if we're not letting it in, because we're obsessed with revenge"
Output: NOT_A_QUESTION

Text: "Yeah, there's energy in that isn't there that you'd like?"
Output: NOT_A_QUESTION

Text: "Can you explain how the mastermind principle actually works in practice?"
Output: How does the mastermind principle work in practice?

Text: "I'm really curious about how faith relates to achieving goals."
Output: How does faith relate to achieving goals?

Text: "If you look at this diagram now that you can see the big circle is the mind"
Output: NOT_A_QUESTION

Text: "Are we halfway or are we about halfway?"
Output: NOT_A_QUESTION

YOUR OUTPUT:"""


def book_batch_analysis_prompt(
    *, items_text: str, taxonomy_categories: list[str] | None = None
) -> str:
    """Prompt for batch LLM analysis of book chunks.

    Args:
        items_text: Formatted text with all chunks to analyze
        taxonomy_categories: Optional list of taxonomy category names

    Returns:
        Formatted prompt string
    """
    category_instruction = ""
    if taxonomy_categories:
        cat_list = ", ".join(taxonomy_categories)
        category_instruction = f"\n- category: Best-matching from: [{cat_list}]"

    return f"""Analyze these book content segments.

SEGMENTS:
{items_text}

For each chunk (CHUNK 0, CHUNK 1, ...), extract:
- topic: Concise 5-8 word summary of the main theme or concept{category_instruction}

Return as TSV, one line per chunk:
index<TAB>topic<TAB>category

Rules:
- Use REAL tab characters (\\t), not "<TAB>"
- topic should capture the core idea or principle discussed
- category can be empty if no taxonomy provided
- Maintain chunk order (0, 1, 2, ...)

Example:
0\tBuilding wealth through persistence\tpersonal_development
1\tThe power of mastermind groups\tmindset
2\tApplying success principles in business\taction_planning"""


def qa_validate_answer_prompt(*, answer_text: str, topic: str) -> str:
    """Prompt to validate if QA answer content is worth indexing.

    Filters out meaningless answers that don't provide educational value.
    """
    return f"""Evaluate if this answer content is WORTH INDEXING for search.

TOPIC: {topic}

ANSWER CONTENT:
{answer_text[:1500]}

TASK: Determine if this content provides educational/instructional value worth indexing.

REJECT (return "NOT_VALUABLE") if the content is:
1. ADMINISTRATIVE/LOGISTICAL:
   - Program logistics: "Let me share my screen", "Can everyone hear me?"
   - Technical issues: "My mic was muted", "Let me fix the audio"
   - Scheduling: "We'll start in a minute", "Let's take a break"

2. PROMOTIONAL/MARKETING:
   - Program pricing/access: "It's a free program", "72 hours access"
   - Upsells: "If you want more, check out our premium..."
   - Testimonial solicitation: "Leave us a review"

3. GREETINGS/CLOSINGS (without substance):
   - Pure greetings: "Welcome everyone!", "Good morning!"
   - Pure sign-offs: "See you next time", "God bless"
   - Only pleasantries without any teaching

4. TOO SHORT/VAGUE:
   - Less than ~50 words of actual content
   - Just acknowledgments: "Right", "Exactly", "Yes"
   - Vague without specifics

5. OFF-TOPIC TANGENTS:
   - Personal anecdotes unrelated to the teaching
   - Completely off-topic discussions

ACCEPT (return "VALUABLE") if the content:
- Teaches a concept, principle, or method
- Provides actionable advice or insights
- Explains an idea that someone might search for
- Contains substantive educational content (even if informal)

OUTPUT FORMAT:
- Return exactly "VALUABLE" or "NOT_VALUABLE"

YOUR OUTPUT:"""


def video_validate_segment_prompt(*, segment_text: str, video_title: str) -> str:
    """Prompt to validate if a video segment is worth indexing.

    Filters out segments that don't provide educational value.
    """
    return f"""Evaluate if this video segment is WORTH INDEXING for search.

VIDEO: {video_title}

SEGMENT CONTENT:
{segment_text[:1500]}

TASK: Determine if this segment provides educational/instructional value worth indexing.

REJECT (return "NOT_VALUABLE") if the segment is:

1. TECHNICAL/META:
   - Audio/video issues: "Can you hear me?", "Let me adjust the camera"
   - Screen sharing: "Let me share my screen", "Can you see this?"
   - Recording notices: "We're recording", "This will be available later"

2. PURE INTRODUCTIONS/OUTROS:
   - Just greetings: "Hey everyone, welcome!"
   - Just sign-offs: "Thanks for watching", "See you next time"
   - (Note: Intros WITH content preview ARE valuable)

3. HOUSEKEEPING:
   - Break announcements: "Let's take a 5 minute break"
   - Scheduling: "Next week we'll cover..."
   - Administrative: "Submit your questions in the chat"

4. TOO SHORT/FRAGMENTED:
   - Incomplete thoughts or mid-sentence fragments
   - Just transitions: "So...", "Now...", "Alright..."
   - Less than ~30 meaningful words
   - Text starting mid-thought (e.g., "matter. You know, we, at one point...")

5. PROMOTIONAL WITHOUT SUBSTANCE:
   - Pure sales pitches
   - Program logistics without teaching

6. GARBLED/TRANSCRIPTION ERRORS:
   - Nonsensical word combinations
   - Obvious speech-to-text errors that make text incomprehensible
   - Sentences that don't make grammatical sense
   - "I was in the middle of a crappy swing. In a two people who should have had dinner..."

ACCEPT (return "VALUABLE") if the segment:
- Explains a concept, principle, or teaching
- Shares an insight, story with a lesson, or practical advice
- Contains searchable educational content
- Even informal teaching counts if substantive
- Is a COMPLETE, coherent thought (not a fragment)

OUTPUT FORMAT:
- Return exactly "VALUABLE" or "NOT_VALUABLE"

YOUR OUTPUT:"""


def qa_rephrase_question_prompt(*, question: str, answer: str) -> str:
    """Prompt to rephrase unclear questions using answer context.

    Evaluates if question is clear as standalone, and if not, rephrases it
    using context from both the question and answer.

    Args:
        question: Current question text (may be unclear/ambiguous)
        answer: Answer text (provides context for clarification)

    Returns:
        Formatted prompt string
    """
    return f"""Rephrase this question to be clear and self-contained.

Evaluate the question:
1. If it's already clear and self-contained (can be understood without context), return it as-is
2. If it's unclear, ambiguous, or references something not in the question itself, rephrase it using context from the answer

QUESTION:
{question}

ANSWER:
{answer[:500]}

Rules:
- Return a clear, self-contained question (max 300 characters)
- If original question is already clear, return it unchanged
- If unclear, use answer context to clarify what is being asked
- Make it specific and actionable
- Remove ambiguous references ("that", "this", "it" without clear referent)
- Keep it natural and conversational

Examples:
Question: "What about that?"
Answer: "The three principles are persistence, clarity, and action."
Rephrased: "What are the three principles?"

Question: "Can you explain more?"
Answer: "The mastermind principle works by combining multiple minds..."
Rephrased: "How does the mastermind principle work?"

Question: "What are the three principles we'll be covering?"
Answer: "We'll cover persistence, clarity, and action."
Rephrased: "What are the three principles we'll be covering?" (already clear)

Return the rephrased question (or original if already clear):
"""
