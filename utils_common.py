import logging

def setup_logger(name: str) -> logging.Logger:
    """
    Set up a logger with a standard format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger
