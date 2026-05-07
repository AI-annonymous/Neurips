from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional



def setup_logger(log_dir: Optional[str] = None, name: str = "neurips_cmi_search") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "run.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@contextmanager
def timed(logger: logging.Logger, message: str) -> Iterator[None]:
    start = time.perf_counter()
    logger.info("[START] %s", message)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("[DONE ] %s | %.2f sec", message, elapsed)
