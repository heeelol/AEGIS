"""
AEGIS Core — Main Entry Point
==============================
Real-time sensor fusion for procedural verification in HMLV kitting.

Usage:
    python -m src.main                              # default config
    python -m src.main --config config/settings.yaml  # custom config
"""

import argparse
import logging
import sys

from src.pipeline import Pipeline

logger = logging.getLogger("aegis")


def main() -> int:
    """Entry point for AEGIS Core."""
    parser = argparse.ArgumentParser(description="AEGIS Core — Kitting Verification System")
    parser.add_argument(
        "--config", "-c",
        default="config/settings.yaml",
        help="Path to YAML configuration file (default: config/settings.yaml)",
    )
    args = parser.parse_args()

    try:
        pipeline = Pipeline(config_path=args.config)
        pipeline.run()
        return 0
    except Exception as e:
        logger.error(f"AEGIS Core failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
