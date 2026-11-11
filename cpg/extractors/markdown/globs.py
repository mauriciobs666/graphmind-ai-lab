"""The prompt-bearing Markdown glob list: read from the manifest file the caller names, never defaulted or duplicated here.

A ``.md`` file is ``prompt-md`` iff it matches an ``include`` glob and no ``except`` glob of a manifest's
``promptBearingMarkdown`` block; every other Markdown file is ``doc``. Production passes the **system** manifest
(``ccg/system.yaml``); the eval manifest (``cpg/eval/eval-manifest.yaml``, ``ccg-eval-manifest-v1``) is an eval/fixture
input and is accepted too, as the census scope (eval-protocol §3.3). The grammar, validator and matcher are
``ccg.globs`` (shared with ``ccg.manifest``); nothing falls back to a built-in list.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ccg.globs import KLASSES, Globs, ManifestError, check_block, _match  # noqa: F401  (re-exported: the public surface)

SCHEMA = "ccg-eval-manifest-v1"
_MAX_MANIFEST_BYTES = 262_144


class _StrictLoader(yaml.SafeLoader):
    """``SafeLoader`` that refuses a duplicate mapping key instead of keeping the last one."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            dup = key in seen
        except TypeError:                                # a list or mapping used as a key
            raise ManifestError("a manifest key must be a scalar") from None
        if dup:
            raise ManifestError("duplicate key in the manifest")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_globs(manifest: str | Path) -> Globs:
    """The prompt-bearing glob list of the manifest at ``manifest`` (required: there is no default file).

    ``ManifestError`` if the file is absent, unreadable, has a duplicate key, is neither a system manifest
    (``schemaVersion`` and ``system``) nor a ``ccg-eval-manifest-v1`` one, or has no valid ``promptBearingMarkdown``."""
    path = Path(manifest)
    try:
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ManifestError(f"{path}: larger than {_MAX_MANIFEST_BYTES} bytes")
        doc = yaml.load(path.read_bytes(), Loader=_StrictLoader)
    except FileNotFoundError as e:
        raise ManifestError(f"manifest absent: {path}") from e
    except (OSError, yaml.YAMLError) as e:
        raise ManifestError(f"manifest unreadable: {path}: {e.__class__.__name__}") from e
    if not isinstance(doc, dict) or not (doc.get("schema") == SCHEMA or ("schemaVersion" in doc and "system" in doc)):
        raise ManifestError(f"{path}: neither a system manifest nor a {SCHEMA} manifest")
    if "promptBearingMarkdown" not in doc:
        raise ManifestError("promptBearingMarkdown is absent from the manifest")
    return check_block(doc["promptBearingMarkdown"])
