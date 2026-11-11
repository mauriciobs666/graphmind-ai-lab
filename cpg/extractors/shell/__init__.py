"""Shell extractor (C-1001 P1-10); see ``extract.py`` and ``README.md``."""
from .extract import MAX_FILE_BYTES, SourceRefused, extract_shell

__all__ = ["extract_shell", "SourceRefused", "MAX_FILE_BYTES"]
