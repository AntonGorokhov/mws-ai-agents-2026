"""Input validation for competition directory."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def check_competition_dir(competition_dir: str, target_column: str) -> dict:
    """Validate competition directory. Returns {"ok": bool, "error": str | None, "info": dict}."""
    d = Path(competition_dir)

    if not d.exists():
        return {"ok": False, "error": f"Directory not found: {d}", "info": {}}

    train_path = d / "train.csv"
    test_path = d / "test.csv"

    if not train_path.exists():
        return {"ok": False, "error": "train.csv not found", "info": {}}
    if not test_path.exists():
        return {"ok": False, "error": "test.csv not found", "info": {}}

    for f in [train_path, test_path]:
        if f.stat().st_size > MAX_FILE_SIZE_BYTES:
            return {"ok": False, "error": f"{f.name} exceeds 2GB", "info": {}}

    train_df = pd.read_csv(train_path, nrows=20)
    if len(train_df) < 10:
        return {"ok": False, "error": "train.csv has fewer than 10 rows", "info": {}}

    if target_column not in train_df.columns:
        return {"ok": False, "error": f"Target column '{target_column}' not in train.csv", "info": {}}

    return {
        "ok": True,
        "error": None,
        "info": {
            "train_columns": list(train_df.columns),
            "train_rows_preview": len(train_df),
            "test_path": str(test_path),
            "train_path": str(train_path),
        },
    }
