import logging
import os
import re
import unicodedata
from pathlib import Path


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _safe_message(value: object) -> str:
    text = str(value or "")
    text = _CONTROL_RE.sub("", text).replace("\r", "\\r").replace("\n", "\\n")
    return text[:2000]


class _SanitizingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _safe_message(record.getMessage())
        record.args = ()
        return True


def build_logger(name: str, log_dir: Path | str | None = None, *, level: str | int | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is None:
        env_level = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sanitizer = _SanitizingFilter()
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logger.level)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(sanitizer)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(sanitizer)
        logger.addHandler(file_handler)

    return logger
