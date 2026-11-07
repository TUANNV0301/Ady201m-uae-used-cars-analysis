"""
Unified Logging Configuration
UAE Used-Car Market Analysis (ADY201m)
Logs all execution steps to both the console and persistent files inside logs/
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from src.config import LOGS_DIR, to_rel_path


def get_logger(
    name: str = "pipeline",
    log_file: Optional[str] = "pipeline.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configures and returns a logger that records events to:
    1. Console (sys.stdout)
    2. Persistent log file inside the logs/ folder
    Ensures safe formatting and no duplicate handlers.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent attaching multiple handlers on repeated calls
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File Handler (stores logs in logs/<log_file>)
        if log_file:
            file_path = LOGS_DIR / log_file
            fh = logging.FileHandler(str(file_path), mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
