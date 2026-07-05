"""Minimal console logging configuration for PlantGeneAnn.

The command-line pipelines log to ``stderr`` only.  The top-level
``PlantGeneAnn`` logger owns the single console handler; child module loggers
propagate to it and must not install their own handlers.
"""

from __future__ import annotations

import logging
import sys


LOGGER_NAME = "PlantGeneAnn"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def setup_logger(*, verbose: bool = False, enabled: bool = True) -> logging.Logger:
    """Configure and return the shared PlantGeneAnn console logger.

    Args:
        verbose: Emit DEBUG messages when true; otherwise emit INFO and above.
        enabled: Install the console handler when true. Accelerate non-main
            ranks pass false so only rank 0 writes pipeline logs.

    Reconfiguration is intentional and idempotent: entry points call this once
    after parsing arguments, while tests and embedded callers may call it more
    than once. Existing handlers are removed and closed before the new handler
    is installed, preventing duplicated messages.
    """

    logger = logging.getLogger(LOGGER_NAME)
    level = logging.DEBUG if verbose else logging.INFO

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(level)
    logger.propagate = False

    if enabled:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        logger.addHandler(console_handler)
    else:
        # A NullHandler also prevents logging.lastResort from printing WARNING
        # messages emitted by child loggers on non-main Accelerate ranks.
        logger.addHandler(logging.NullHandler())

    return logger
