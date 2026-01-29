"""
Privacy utilities - MUST be imported before any other modules.

This module disables telemetry and analytics across all dependencies.
For therapy applications, user privacy is paramount.
"""

import logging
import os

# ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"

# Hugging Face / Sentence Transformers telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "0"  # Don't force offline, but disable telemetry
os.environ["DO_NOT_TRACK"] = "1"  # General opt-out standard

# Disable various analytics loggers
TELEMETRY_LOGGERS = [
    "chromadb.telemetry",
    "chromadb.telemetry.product.posthog",
    "posthog",
    "httpx",
    "huggingface_hub",
    "sentence_transformers",
]


def disable_telemetry() -> None:
    """Disable all telemetry. Call this at application startup."""
    for logger_name in TELEMETRY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)


# Auto-disable on import
disable_telemetry()
