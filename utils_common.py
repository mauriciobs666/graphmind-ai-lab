import contextvars
import logging
import os
from pathlib import Path
from collections import OrderedDict
from typing import Optional

from config import Config

_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
_LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - session=%(session_id)s - %(message)s"
_SESSION_ID = contextvars.ContextVar("session_id", default=None)
_SESSION_HANDLER_LIMIT = int(os.getenv("LOG_SESSION_HANDLER_LIMIT", "100"))
_SESSION_HANDLERS: "OrderedDict[str, logging.Handler]" = OrderedDict()
_LOG_EXCLUDE_PREFIXES = tuple(
    prefix.strip()
    for prefix in os.getenv("LOG_EXCLUDE_PREFIXES", "watchdog,streamlit,httpcore").split(",")
    if prefix.strip()
)

_base_record_factory = logging.getLogRecordFactory()
def _record_factory(*args, **kwargs):
    record = _base_record_factory(*args, **kwargs)
    session_id = _SESSION_ID.get()
    record.session_id = session_id if session_id else "-"
    return record

if not getattr(logging, "_graphmind_record_factory_set", False):
    logging.setLogRecordFactory(_record_factory)
    logging._graphmind_record_factory_set = True

def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

def _handler_has_path(handler: logging.Handler, path: Path) -> bool:
    if not isinstance(handler, logging.FileHandler):
        return False
    return Path(getattr(handler, "baseFilename", "")).resolve() == path.resolve()

def _ensure_app_log_handler() -> None:
    _ensure_log_dir()
    root = logging.getLogger()
    app_log = _LOG_DIR / "app.log"
    for handler in root.handlers:
        if _handler_has_path(handler, app_log):
            return
    handler = logging.FileHandler(app_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_AppLogFilter())
    if _LOG_EXCLUDE_PREFIXES:
        handler.addFilter(_ExcludeLoggerFilter(_LOG_EXCLUDE_PREFIXES))
    root.addHandler(handler)

def _ensure_root_level() -> None:
    level_name = Config.get_log_level().upper()
    level = logging.getLevelName(level_name)
    root = logging.getLogger()
    root.setLevel(level if isinstance(level, int) else logging.DEBUG)

class _SessionFilter(logging.Filter):
    def __init__(self, session_id: str):
        super().__init__()
        self.session_id = session_id

    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "session_id", None) == self.session_id

class _AppLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, "session_id", "-") == "-"

class _ExcludeLoggerFilter(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]):
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(self.prefixes)

def set_active_session(session_id: Optional[str]) -> contextvars.Token:
    return _SESSION_ID.set(session_id)

def reset_active_session(token: contextvars.Token) -> None:
    _SESSION_ID.reset(token)

def ensure_session_log_handler(session_id: str) -> None:
    if not session_id:
        return
    _ensure_log_dir()
    root = logging.getLogger()
    session_log = _LOG_DIR / f"session_{session_id}.log"
    existing = _SESSION_HANDLERS.get(session_id)
    if existing and _handler_has_path(existing, session_log):
        return
    handler = logging.FileHandler(session_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_SessionFilter(session_id))
    if _LOG_EXCLUDE_PREFIXES:
        handler.addFilter(_ExcludeLoggerFilter(_LOG_EXCLUDE_PREFIXES))
    root.addHandler(handler)
    _SESSION_HANDLERS[session_id] = handler
    _SESSION_HANDLERS.move_to_end(session_id)
    while len(_SESSION_HANDLERS) > _SESSION_HANDLER_LIMIT:
        old_session, old_handler = _SESSION_HANDLERS.popitem(last=False)
        root.removeHandler(old_handler)
        try:
            old_handler.close()
        except Exception:
            pass


def format_currency(value: float) -> str:
    return f"R${value:.2f}".replace(".", ",")

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with a standard format.
    """
    _ensure_app_log_handler()
    _ensure_root_level()
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    level_name = Config.get_log_level().upper()
    level = logging.getLevelName(level_name)
    if isinstance(level, int):
        logger.setLevel(level)
    else:
        logger.setLevel(logging.DEBUG)
    return logger
