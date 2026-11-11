"""Prompt/Markdown extractor (P1-12). Spec: bundle-spec3 §4.4 ("Markdown prompts"), §5.2 (``file``), §5.6 (``callsite``,
``mention``), §7 (``prompt-mention``); plan ``docs/plans/connected-code-graph-p1-extraction.md`` § P1-12.

Every Markdown file gets a ``file`` record (``lang`` markdown, ``kind`` prompt) and one ``path::<document>`` symbol
whose span is the whole file. Its class comes from the eval manifest's globs (``globs.py``):

- ``doc``: ``role`` doc, no call sites (documentation snippets are not callers).
- ``prompt-md``: ``role`` primary (``example`` under the spec's example rule), plus one call site per *mention* of
  a tool, script or command, ``callKind`` mention, ``possible``/``prompt-mention``, never ``confirmed`` (a prose
  mention proves nothing, and the linker, not this module, decides what it names).

A mention is, outside fenced code (a fence is not a call, whatever it says):

- **tool**: an ``mcp__<server>__<tool>`` name, in prose, in an inline code span or in front matter (where an agent
  declares the tools it uses);
- **script**: a token of an inline code span that is a path or name ending ``.sh .py .js .mjs .ts``, as written;
- **bare tool name**: a word in ``known_tools`` (the caller's vocabulary; empty switches this off), on word boundaries
  (``[A-Za-z0-9_]`` runs) in prose, inline spans and table cells, never in fenced code or front matter;
- **command**: the first word of an inline code span when it is one of ``COMMANDS`` (a closed list of CLI launchers),
  unless the span names a script (``bash x.sh`` mentions ``x.sh``, not ``bash``).

A front-matter ``disallowedTools:`` entry (a deny-list) is never a mention; ``tools:`` entries are.

Not recognised, by design: indented code blocks (indistinguishable from list continuation without a full CommonMark
parser), spans across lines, link targets. The file is untrusted text: it is only decoded and scanned, never executed
or followed; no regular expression here can backtrack beyond linear in a capped line; a line is scanned up to
``LINE_CAP`` characters, a file up to ``MAX_FILE_BYTES`` (larger raises ``MarkdownError``: the caller skips it) and
``MAX_MENTIONS`` call sites.

This module writes none of the fields guarded by ``extractors/common/writer_rules.py`` (``command``, ``args``,
``bindValue``, ``default``, ``valueHash``, ``defaultHash``): a mention carries only its name.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

from ccg import ids

from .globs import Globs

MAX_FILE_BYTES = 2_097_152
LINE_CAP = 4000
MAX_MENTIONS = 1000
NAME_CAP = 200

#: CLI launchers a prompt may tell its reader to run. A closed list: the extractor never guesses at an open "command".
COMMANDS = frozenset({
    "docker", "docker-compose", "redis-cli", "curl", "git", "npm", "npx", "pnpm", "uv", "uvx", "pip", "pytest",
    "make", "claude", "opencode", "kiro-cli", "joern", "bash", "sh", "python", "python3", "node",
})
_SCRIPT_EXT = (".sh", ".py", ".js", ".mjs", ".ts")

_MCP = re.compile(r"mcp__[A-Za-z0-9][A-Za-z0-9-]{0,63}__[A-Za-z0-9_][A-Za-z0-9_-]{0,63}")
_SCRIPT = re.compile(r"(?:\.{1,2}/)?(?:[A-Za-z0-9_.-]{1,80}/){0,8}[A-Za-z0-9_-][A-Za-z0-9_.-]{0,80}")
_WORD = re.compile(r"[A-Za-z0-9_]+")
_FM_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}:")
FRONT_MATTER_MAX_LINES = 200
_EXAMPLE_BASE = re.compile(r"(?:.*\.example.*|.*\.sample.*|example.*|sample.*)", re.S)


class MarkdownError(ValueError):
    """The file cannot be extracted (over ``MAX_FILE_BYTES``): the caller skips it."""


@dataclass
class Extraction:
    klass: str                                  # ``prompt-md`` or ``doc``
    records: list[dict]                         # the ``file`` record, the ``<document>`` symbol, then the call sites
    mentions_capped: bool = False
    lines_truncated: int = 0                    # lines scanned only up to ``LINE_CAP`` characters
    names: list[str] = field(default_factory=list)   # the mention names, in order (convenience for callers/tests)


def _strip_container(line: str) -> str:
    """Drop the block-container prefixes a fence may sit behind: indentation, ``>`` and one list marker per level."""
    s = line
    for _ in range(8):
        t = s.lstrip(" \t")
        if t.startswith(">"):
            s = t[1:]
        elif t[:2] in ("- ", "* ", "+ "):
            s = t[2:]
        else:
            m = t.split(" ", 1)[0]
            if 1 < len(m) <= 10 and m[:-1].isdigit() and m[-1] in ".)" and len(t) > len(m):
                s = t[len(m) + 1:]
            else:
                return t
    return s.lstrip(" \t")


def _fence_run(line: str) -> tuple[str, int, str] | None:
    """(char, length, rest) when the line opens or closes a fence: >= 3 backticks or tildes after container prefixes."""
    t = _strip_container(line)
    if not t or t[0] not in "`~":
        return None
    ch = t[0]
    n = len(t) - len(t.lstrip(ch))
    if n < 3:
        return None
    return ch, n, t[n:]


def _spans(line: str):
    """Single-line inline code spans: (content, column of content). A span closes on a run of exactly the opening length."""
    i, n = 0, len(line)
    while i < n:
        if line[i] != "`":
            i += 1
            continue
        j = i
        while j < n and line[j] == "`":
            j += 1
        k, width = j, j - i
        found = -1
        while k < n:
            if line[k] != "`":
                k += 1
                continue
            e = k
            while e < n and line[e] == "`":
                e += 1
            if e - k == width:
                found = k
                break
            k = e
        if found < 0:
            i = j
            continue
        yield line[j:found], j
        i = found + width


def _is_script(token: str) -> bool:
    return (token.lower().endswith(_SCRIPT_EXT) and len(token) <= NAME_CAP and _SCRIPT.fullmatch(token) is not None)


def _span_mentions(content: str, col: int):
    """(name, column) for the script and command mentions of one inline code span."""
    text = content.strip()
    if not text or len(text) > 300:
        return
    base = col + (len(content) - len(content.lstrip()))
    if text.startswith("$ "):
        base += 2
        text = text[2:].lstrip()
    # tokens (at most the first 6) with columns
    tokens: list[tuple[str, int]] = []
    pos = 0
    while len(tokens) < 6 and pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        s = pos
        while pos < len(text) and not text[pos].isspace():
            pos += 1
        if pos > s:
            tokens.append((text[s:pos], base + s))
    scripts = [(t, c) for t, c in tokens if _is_script(t)]
    if scripts:
        yield from scripts
        return
    if tokens and tokens[0][0] in COMMANDS:
        yield tokens[0]


def _check_path(name: str, p: str) -> None:
    """A repository-relative POSIX path in normal form; anything else would silently classify ``doc`` (F5)."""
    if (not p or p.startswith("/") or "\\" in p or "\x00" in p
            or any(seg in ("", ".", "..") for seg in p.split("/"))):
        raise MarkdownError(f"{name} {p!r} is not a normalised repository-relative POSIX path")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _example(path: str) -> bool:
    segs = path.split("/")
    base = segs[-1].lower()
    return (_EXAMPLE_BASE.fullmatch(base) is not None) or any(s in ("examples", "samples") for s in segs[:-1])


def _front_matter_end(lines: list[str]) -> int:
    """The 1-based line of the closing ``---`` of a leading front matter block, else 0. The block must open with
    ``---``, its first non-blank line must look like YAML (``key:``, a ``-`` list item or a ``#`` comment: otherwise
    the ``---`` is a horizontal rule) and close within ``FRONT_MATTER_MAX_LINES`` lines."""
    if not lines or lines[0].rstrip() != "---":
        return 0
    for n in range(2, min(len(lines), FRONT_MATTER_MAX_LINES) + 1):
        text = lines[n - 1].rstrip()
        if text == "---":
            return n if _yaml_first(lines[1:n - 1]) else 0
    return 0


def _yaml_first(body: list[str]) -> bool:
    for line in body:
        t = line.strip()
        if t:
            return bool(_FM_KEY.match(t)) or t.startswith("#") or t == "-" or t.startswith("- ")
    return False


def _bare_ok(line: str, start: int, end: int) -> bool:
    """Word boundary for a vocabulary name: not glued to ``-`` or a non-ASCII letter/digit, and not followed by
    ``.`` plus an alphanumeric (a file name or domain); a sentence-final ``.`` or ``,`` still matches."""
    if start > 0 and (line[start - 1] == "-" or (line[start - 1].isalnum() and not line[start - 1].isascii())):
        return False
    if end < len(line):
        c = line[end]
        if c == "-" or (c.isalnum() and not c.isascii()):
            return False
        if c == "." and end + 1 < len(line) and line[end + 1].isalnum():
            return False
    return True


def _scan(lines: list[str], capped: list[int], known: frozenset[str]):
    """The mentions ``(line, column, name)`` of an already-split file, in source order, outside fences."""
    out: list[tuple[int, int, str]] = []
    fence: tuple[str, int] | None = None
    fm_end = _front_matter_end(lines)
    key = ""
    for n, raw in enumerate(lines, start=1):
        line = raw
        if fm_end and n <= fm_end:
            if n in (1, fm_end):
                continue
            m = _FM_KEY.match(line)
            if m:
                key = m.group()[:-1].lower()
            if key == "disallowedtools":
                continue
            out.extend((n, mm.start(), mm.group()) for mm in _MCP.finditer(line[:LINE_CAP]))
            continue
        if len(line) > LINE_CAP:
            line = line[:LINE_CAP]
            capped[0] += 1
        run = _fence_run(line)
        if fence is not None:
            if run and run[0] == fence[0] and run[1] >= fence[1] and not run[2].strip():
                fence = None
            continue
        if run and not (run[0] == "`" and "`" in run[2]):
            fence = (run[0], run[1])
            continue
        found = [(m.start(), m.group()) for m in _MCP.finditer(line)]
        for content, col in _spans(line):
            found.extend((c, t) for t, c in _span_mentions(content, col))
        if known:
            found.extend((w.start(), w.group()) for w in _WORD.finditer(line) if w.group() in known and _bare_ok(line, w.start(), w.end()))
        for col, name in sorted(set(found)):
            out.append((n, col, name))
    return out


def extract_markdown(repo_path: str, data: bytes, *, repo_id: str, unit_id: str, globs: Globs,
                     bundle_path: str | None = None, history: tuple[str, str] | None = None,
                     known_tools: frozenset[str] = frozenset()) -> Extraction:
    """Records for one Markdown file.

    ``repo_path`` is the repository-relative POSIX path the manifest's globs are matched against; ``bundle_path`` (default
    the same) is the ``path`` the records carry (relative to the unit's root when the unit is not the repository).
    ``known_tools`` is the vocabulary of bare tool names (P1-44's; empty: off). ``history`` is ``(lastChangedAt, lastChangedCommit)`` for the ``<document>`` symbol, from the assembler's blame."""
    _check_path("repo_path", repo_path)
    if bundle_path is not None:
        _check_path("bundle_path", bundle_path)
    if len(data) > MAX_FILE_BYTES:
        raise MarkdownError(f"{repo_path}: {len(data)} bytes exceeds the {MAX_FILE_BYTES}-byte cap")
    path = bundle_path or repo_path
    klass = globs.classify(repo_path)
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [x[:-1] if x.endswith("\r") else x for x in lines]
    line_count = len(lines)

    fid = ids.file_id(repo_id, path)
    role = "doc" if klass == "doc" else ("example" if _example(path) else "primary")
    qname = f"{path}::<document>"
    sid = ids.symbol_id("markdown", repo_id, qname)
    fingerprint = hashlib.sha256("\n".join(s for s in (x.strip() for x in lines) if s).encode("utf-8")).hexdigest()
    records: list[dict] = [
        {"type": "file", "id": fid, "repoId": repo_id, "unitId": unit_id, "path": path, "lang": "markdown",
         "kind": "prompt", "role": role, "contentHash": hashlib.sha256(data).hexdigest(), "lineCount": line_count},
    ]
    symbol = {"type": "symbol", "id": sid, "lang": "markdown", "repoId": repo_id, "qname": qname, "fileId": fid,
              "path": path, "kind": "document", "name": "<document>", "nameNorm": "<document>", "lineStart": 1,
              "lineEnd": max(line_count, 1), "fingerprint": fingerprint, "complexity": 1}
    if history is not None:
        symbol["lastChangedAt"], symbol["lastChangedCommit"] = history
    records.append(symbol)
    ex = Extraction(klass=klass, records=records)
    if klass == "doc":
        return ex

    capped = [0]
    mentions = _scan(lines, capped, known_tools)
    ex.lines_truncated = capped[0]
    if len(mentions) > MAX_MENTIONS:
        mentions = mentions[:MAX_MENTIONS]
        ex.mentions_capped = True
    seen: dict[str, int] = {}
    for line, _col, name in mentions:         # source order is the (line, column) order of §5.7
        name = _norm(name)
        ordinal = seen.get(name, 0)
        seen[name] = ordinal + 1
        records.append({"type": "callsite", "id": ids.callsite_id(sid, name, ordinal), "symbolId": sid,
                        "calleeName": name, "ordinal": ordinal, "callKind": "mention", "line": line,
                        "resolution": "unresolved", "targets": [], "status": "possible", "reason": "prompt-mention"})
        ex.names.append(name)
    return ex
