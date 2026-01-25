import logging

from config import Config


def format_currency(value: float) -> str:
    return f"R${value:.2f}".replace(".", ",")

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with a standard format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    level_name = Config.get_log_level().upper()
    level = logging.getLevelName(level_name)
    if isinstance(level, int):
        logger.setLevel(level)
    else:
        logger.setLevel(logging.DEBUG)
    return logger
