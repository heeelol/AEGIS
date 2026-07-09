"""
Centralized logging setup for AEGIS Core.
"""

import logging
from pathlib import Path


def setup_logger(name: str = "aegis", config: dict | None = None) -> logging.Logger:
    """Create a logger with console + optional file output."""
    log_cfg = config.get("logging", {}) if config else {}
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    output_dir = log_cfg.get("output_dir")
    if output_dir:
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "aegis.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
