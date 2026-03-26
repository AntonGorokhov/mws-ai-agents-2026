"""Agent action logging."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(workspace_dir: str, level: int = logging.INFO):
    """Configure logging to console and file."""
    log_dir = Path(workspace_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    # Console handler (with flush)
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    console.addFilter(lambda record: console.stream.flush() or True)
    root.addHandler(console)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    root.addHandler(fh)

    logger.info("Logging to %s", log_file)


def save_agent_logs(agent_logs: list[dict], workspace_dir: str):
    """Save all agent logs to a JSON file."""
    log_dir = Path(workspace_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "agent_logs.json"
    log_file.write_text(json.dumps(agent_logs, indent=2, default=str))
    logger.info("Agent logs saved to %s", log_file)
