"""LLM output sanitization and retry logic."""

import re
import logging

logger = logging.getLogger(__name__)


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response (handles markdown fences)."""
    # Try to find ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try generic ``` blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no fences found, return the raw text (might be pure code)
    return text.strip()


def sanitize_llm_output(text: str) -> str:
    """Remove think tags and other noise from LLM output."""
    # Remove <think>...</think> blocks (DeepSeek R1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()
