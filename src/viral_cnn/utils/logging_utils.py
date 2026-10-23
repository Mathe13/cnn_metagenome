"""
Path and artifact logging utilities for traceable experiment execution.

Rule (AGENTS.md):
- Whenever dataloaders, datasets (e.g. Parquet, CSV), models or checkpoints are loaded,
  the source path must be logged explicitly.
- Enabled by default, can be disabled by setting the environment variable LOG_PATH to '0', 'false', or 'no'.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

logger = logging.getLogger("viral_cnn.paths")


def is_path_logging_enabled() -> bool:
    """
    Checks if path logging is enabled.
    Defaults to True unless LOG_PATH is set to '0', 'false', 'no', 'f', 'off'.
    """
    val = os.environ.get("LOG_PATH", "true").strip().lower()
    return val not in ("0", "false", "no", "f", "off", "disable", "disabled")


def log_artifact_path(
    category: str,
    path: Union[str, Path, Sequence[Union[str, Path]]],
    extra: Optional[str] = None,
) -> None:
    """
    Logs the source path of an artifact (checkpoint, dataloader, taxonomy file, etc.).
    Prints to stdout for interactive notebook visibility and records via standard logging.
    """
    if not is_path_logging_enabled():
        return

    if isinstance(path, (list, tuple)):
        if len(path) == 0:
            formatted_path = "[]"
        elif len(path) == 1:
            formatted_path = str(path[0])
        else:
            formatted_path = f"{path[0]} ... ({len(path)} files total)"
    else:
        formatted_path = str(path)

    msg = f"[LOG_PATH] Loading {category} from: {formatted_path}"
    if extra:
        msg += f" | {extra}"

    print(msg)
    logger.info(msg)

