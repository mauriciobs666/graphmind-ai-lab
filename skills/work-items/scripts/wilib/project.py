"""``wi init`` without the key claim (plan 3.9)."""
import json
from pathlib import Path

from . import rules
from .store import Store

GITATTR_LINE = ".work-items/registry.jsonl merge=union"
_SEEDS = ((1, "Defects"), (2, "Kaizen"))   # I01 Defects, I02 Kaizen (FR-14)


class InitError(RuntimeError):
    """init cannot proceed (project already initialized under another key)."""


def _requirements_documents(root):
    found = set(root.glob("docs/requirements/*.md")) | set(root.glob("*/docs/requirements/*.md"))
    return sorted(p.relative_to(root).as_posix() for p in found)


def _ensure_gitattributes(root):
    path = root / ".gitattributes"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if GITATTR_LINE in text.splitlines():
        return
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + GITATTR_LINE + "\n", encoding="utf-8")


def init(root, key, components):
    """Create ``.work-items/`` for ``key``; idempotent for the same key."""
    root = Path(root)
    rules.format_id(key, [])            # validates the key (raises MalformedId)
    wi = root / ".work-items"
    pj_path = wi / "project.json"
    if pj_path.exists():
        existing = json.loads(pj_path.read_text(encoding="utf-8")).get("key")
        if existing != key:
            raise InitError(f"project already initialized with key {existing!r}")
    else:
        wi.mkdir(parents=True, exist_ok=True)
        pj_path.write_text(json.dumps(
            {"key": key, "components": list(components), "legacy_namespaces": {}},
            indent=2) + "\n", encoding="utf-8")
        (wi / "baseline.txt").write_text(
            "".join(p + "\n" for p in _requirements_documents(root)), encoding="utf-8")
    _ensure_gitattributes(root)
    store = Store(root, key)
    with store.transaction() as tx:
        for num, title in _SEEDS:
            id_ = rules.format_id(key, [rules.Segment("I", num)])
            if id_ not in tx.state.items:
                tx.append({"ev": "new", "id": id_, "type": "I", "parent": None,
                           "home": "repo", "title": title})
    store.flush()


def load_namespaces(root):
    """Declared legacy namespaces: ``{"K-": [home, ...], "C-": [...]}`` (plan 3.6)."""
    path = Path(root) / ".work-items" / "project.json"
    data = json.loads(path.read_text(encoding="utf-8")).get("legacy_namespaces") or {}
    return {prefix: list(homes) for prefix, homes in data.items()}
