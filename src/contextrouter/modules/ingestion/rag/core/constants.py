"""Named constants for ingestion pipeline.

Centralizes magic numbers and thresholds for maintainability.
All values can be overridden via settings.toml when needed.
"""

# QA Processing
MIN_STANDALONE_WORDS = 5  # Minimum words for a speaker turn to be standalone
QA_CHUNK_MIN_CHARS = 400  # Minimum characters per QA chunk
QA_CHUNK_MAX_CHARS = 1500  # Maximum characters per QA chunk

# Video Processing
VIDEO_PAUSE_THRESHOLD_S = 0.8  # Pause duration (seconds) to break sentence
VIDEO_SLIDING_WINDOW_SIZE = 3  # Number of sentences per window

# Batch Processing
DEFAULT_BATCH_SIZE = 50
SUMMARY_BATCH_SIZE = 15
VALIDATION_BATCH_SIZE = 50

# LLM Settings
LLM_SUCCESS_RATE_THRESHOLD = 0.8  # Retry if success rate below this
LLM_MAX_BATCH_RETRIES = 3

# Content Validation
MIN_CONTENT_WORDS = 15  # Minimum words for valuable content
MAX_QUOTE_LENGTH = 300  # Maximum chars for quote in validation prompt
