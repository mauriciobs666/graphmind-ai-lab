"""`ManifestMarkers`: the gateway's scope-marker source for the `ccg:` scope prefix (docs/plans/llm-gateway.md §3.6).

The local-only flag has one copy: the installed manifest (`/etc/llmgateway/ccg/system.yaml` in production, a
root-owned byte copy of the tracked one). It is read, owner-checked and parsed on **every** `marker` call; nothing is
cached, so a tightened manifest applies on the next call. Anything the source cannot say raises `MarkerUnreadable`,
which the gateway's policy treats as "refuse hosted" (FR-3). Scope ids:

    ccg:system:<system id>     restricted when any repository of the system is local-only (stickiness)
    ccg:repo:<normalized id>   restricted when that repository is local-only

The broker's own R5/R6 decision over its own copy of the manifest stays authoritative; this source only stops the
gateway choosing a hosted step early.
"""
from __future__ import annotations

import os
from collections.abc import Collection

import yaml

from ccg.manifest import normalize_repo_id
from llmgateway.ownercheck import read_owner_file

DEFAULT_MANIFEST = "/etc/llmgateway/ccg/system.yaml"
PREFIX = "ccg:"
SYSTEM_PREFIX = PREFIX + "system:"
REPO_PREFIX = PREFIX + "repo:"
RESTRICTED = "restricted"
UNRESTRICTED = "unrestricted"
_EGRESS = {"local-only": RESTRICTED, "hosted": UNRESTRICTED}


class MarkerUnreadable(Exception):
    """The installed manifest cannot answer for this scope."""


class _StrictLoader(yaml.SafeLoader):
    """`SafeLoader` refusing a duplicate mapping key (the broker's loader does the same)."""


def _mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False):
    keys = [loader.construct_object(k, deep=True) for k, _ in node.value]
    try:
        if len(set(keys)) != len(keys):
            raise MarkerUnreadable("duplicate key in the manifest")
    except TypeError:
        raise MarkerUnreadable("a manifest key is not a scalar") from None
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def repo_scope(repo_id: str) -> str:
    return REPO_PREFIX + normalize_repo_id(repo_id)


def system_scope(system: str) -> str:
    return SYSTEM_PREFIX + system


class ManifestMarkers:
    """`path` is the installed manifest; `trusted_uids` the uids allowed to own it and its ancestors (production
    `{0}`); `anchor` is `ownercheck.read_owner_file`'s test seam."""

    def __init__(
        self,
        path: str | os.PathLike[str] = DEFAULT_MANIFEST,
        trusted_uids: Collection[int] = frozenset({0}),
        *,
        anchor: str | os.PathLike[str] | None = None,
    ) -> None:
        self._path, self._trusted, self._anchor = os.fspath(path), frozenset(trusted_uids), anchor

    def _egress_by_repo(self) -> tuple[str, dict[str, str]]:
        res = read_owner_file(self._path, self._trusted, anchor=self._anchor)
        if not res.ok:
            raise MarkerUnreadable(f"installed manifest is not owner-protected: {res.reason}")
        try:
            doc = yaml.load(res.data, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
        except yaml.YAMLError:
            raise MarkerUnreadable("installed manifest is not valid YAML") from None
        if not isinstance(doc, dict) or not isinstance(doc.get("system"), str) or not isinstance(doc.get("repos"), list):
            raise MarkerUnreadable("installed manifest has no system and repos")
        out: dict[str, str] = {}
        for repo in doc["repos"]:
            policy = repo.get("policy") if isinstance(repo, dict) else None
            mode = policy.get("egress") if isinstance(policy, dict) else None
            name = repo.get("repoId") if isinstance(repo, dict) else None
            if not isinstance(name, str) or not isinstance(mode, str) or mode not in _EGRESS:
                raise MarkerUnreadable("installed manifest has an ill-typed repository entry")
            out[normalize_repo_id(name)] = _EGRESS[mode]
        return doc["system"], out

    def marker(self, scope_id: str) -> str:
        system, repos = self._egress_by_repo()
        if scope_id.startswith(SYSTEM_PREFIX):
            if scope_id != system_scope(system):
                raise MarkerUnreadable("scope names a system the installed manifest does not describe")
            return RESTRICTED if RESTRICTED in repos.values() else UNRESTRICTED
        if scope_id.startswith(REPO_PREFIX):
            repo = scope_id[len(REPO_PREFIX):]
            if repo not in repos:
                raise MarkerUnreadable("scope names a repository the installed manifest does not list")
            return repos[repo]
        raise MarkerUnreadable("scope id is not under a ccg scope prefix")
