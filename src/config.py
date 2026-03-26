"""Configuration: models, paths, constants."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1/chat/completions"

MODELS = {
    "coder": "qwen/qwen3-coder",
    "coder_fallback": "deepseek/deepseek-v3.2",
    "reasoning": "deepseek/deepseek-v3.2",
    "reasoning_fallback": "deepseek/deepseek-r1",
    "embedding": "all-MiniLM-L6-v2",
}

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
CHROMA_DB_DIR = KNOWLEDGE_BASE_DIR / "chroma_db"
KNOWLEDGE_SOURCES_DIR = KNOWLEDGE_BASE_DIR / "sources"

# Limits
MAX_ITERATIONS = 3
CODE_TIMEOUT_S = 300
MAX_RETRIES = 2
MAX_TOKEN_BUDGET = 200_000
MAX_CODE_FAILURES = 3

# Sandbox blocked patterns
BLOCKED_PATTERNS = ["os.system", "subprocess", "shutil.rmtree", "rm -rf", "__import__"]
