"""``launchFact`` derivations (F-6, P1-11): what a config file says about how something is launched or where it listens.

Spec: `docs/plans/connected-code-graph-bundle-spec3.md` §1.6. This module only *derives drafts* from a parsed tree; the record
fields the writer rules own (``command``, ``args``, ``default``, ``defaultHash``) are written by ``extract`` through
``extractors.common.writer_rules``, which is why a draft names them ``argv`` and ``default_text``. Nothing here executes,
expands or interpolates: a shell-form command is only split, and ``${VAR:-x}`` is only read.

| source | draft |
|---|---|
| Dockerfile ``ENTRYPOINT``/``CMD`` (final stage, last of each) | ``command``, origin ``explicit`` |
| Dockerfile ``EXPOSE`` | ``port`` (protocol default ``tcp``), origin ``explicit`` |
| Compose ``ports[i]`` | ``port`` for the published port (when numeric and different) then the container port; origin ``compose`` |
| Compose ``expose[i]`` | ``port``, origin ``compose`` |
| Compose ``entrypoint``/``command`` | ``command``, origin ``compose``, ``service`` |
| Compose ``build.context`` (or ``build:``) | ``mount`` with ``mountPath`` the relative path as written, origin ``compose``, ``service`` |
| ``.mcp.json`` / Kiro agent ``mcpServers.<s>.command`` + ``args`` | ``command``, origin ``explicit``, ``service`` is the server name |
| any scalar holding ``${VAR:-default}`` or ``${VAR-default}`` | ``env-default``, origin ``env-default`` |

The ``uvicorn --port`` case is a ``command`` whose ``args`` hold the flag (the linker reads the port from the gated argv);
framework defaults are the packs', never the bundle's.
"""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

from ccg import bundle

from . import formats as F

MAX_ARGV = 256
MAX_ARG_CHARS = 4096
MAX_ENV_PER_TEXT = 16


@dataclass(frozen=True)
class Draft:
    kind: str                       # command | port | env-default | mount
    line: int
    origin: str                     # explicit | env-default | compose
    service: str | None = None
    port: int | None = None
    protocol: str | None = None
    mount_path: str | None = None
    env_var: str | None = None
    default_text: str | None = None
    argv: tuple[str, ...] | None = None
    program_known: bool = True      # False: the list is arguments for a program defined elsewhere (CMD beside ENTRYPOINT, Compose command beside entrypoint)


_PORT = re.compile(r"[1-9][0-9]{0,4}")
_PROTOS = ("tcp", "udp", "sctp")
_ENV_DEFAULT = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):?-([^}]*)\}")
_SAFE_CONTEXT = re.compile(r"[A-Za-z0-9._/~+-]+")


def valid_port(text: str) -> int | None:
    return int(text) if _PORT.fullmatch(text) and int(text) <= 65535 else None


def env_defaults(text: str, line: int) -> list[Draft]:
    """``${VAR:-8200}`` and ``${VAR-8200}`` occurrences in ``text`` (not interpolated: read). The default is the text up to the
    first ``}``, so a nested ``${B}`` leaves a ``$`` in it, which the value grammar then withholds."""
    return [Draft("env-default", line, "env-default", env_var=m.group(1), default_text=m.group(2))
            for m in _ENV_DEFAULT.finditer(text[:65536])][:MAX_ENV_PER_TEXT]


def _service_name(name: str) -> str | None:
    """The Compose service (or MCP server) name for the optional ``service`` field: NFC and stripped (§8.1 identity strings; the
    validator rejects anything else), absent if empty or if it would not pass the screen."""
    name = F.norm_name(name)
    return name if name and not bundle.screen_fails(name) else None


def _argv(elements: list[str]) -> tuple[str, ...] | None:
    if not elements or len(elements) > MAX_ARGV or not elements[0] or any(len(e) > MAX_ARG_CHARS for e in elements):
        return None
    return tuple(elements)


def _split_shell(text: str) -> list[str] | None:
    try:
        return shlex.split(text, posix=True)
    except ValueError:
        return None


def _scalar_list(node: F.Node) -> list[str] | None:
    if node.kind != "l" or any(c.kind != "s" for c in node.items):
        return None
    return [c.text for c in node.items]


_child = F.child_node


# --- Dockerfile ------------------------------------------------------------------------------------------------

def dockerfile_drafts(text: str) -> list[Draft]:
    st = F.dockerfile_stage(F.dockerfile_instructions(text))
    out: list[Draft] = []
    for token, line in st.expose:
        m = re.fullmatch(r"([0-9]+)(?:/([A-Za-z]+))?", token)
        port = valid_port(m.group(1)) if m else None
        if port is not None:
            proto = (m.group(2) or "tcp").lower()
            out.append(Draft("port", line, "explicit", port=port, protocol=proto if proto in _PROTOS else "tcp"))
    for ins in (st.entrypoint, st.cmd):
        argv = _argv(F.dockerfile_argv(ins) or [])
        if ins is not None and argv:
            known = not (ins is st.cmd and st.entrypoint is not None)       # a CMD beside an ENTRYPOINT is its arguments
            out.append(Draft("command", ins.line, "explicit", argv=argv, program_known=known))
    return out


# --- Compose ---------------------------------------------------------------------------------------------------

def parse_port_spec(spec: str) -> list[tuple[int, str]]:
    """A Compose short-syntax port ``[ip:][published:]container[/proto]`` as (port, protocol): the published port first when it
    is numeric and differs from the container's. A range, an interpolated part or a malformed spec contributes nothing for that part."""
    body, _, proto = spec.strip().partition("/")
    proto = proto.lower() if proto.lower() in _PROTOS else "tcp"
    body = re.sub(r"^\[[0-9a-fA-F:.]*\]:", "", body)
    parts = body.split(":")
    container = valid_port(parts[-1])
    published = valid_port(parts[-2]) if len(parts) >= 2 else None
    out: list[tuple[int, str]] = []
    if published is not None and published != container:
        out.append((published, proto))
    if container is not None:
        out.append((container, proto))
    return out


def _long_port(el: F.Node, service: str | None) -> list[Draft]:
    """Long syntax ``{target, published, protocol}``: the published port (when it differs) then the target."""
    def port(key: str) -> tuple[int | None, int]:
        node = _child(el, key)
        return (valid_port(node.text) if node is not None and node.kind == "s" else None), (node.line if node else el.line)

    proto = _child(el, "protocol")
    pr = proto.text.lower() if proto is not None and proto.kind == "s" and proto.text.lower() in _PROTOS else "tcp"
    (target, t_line), (published, p_line) = port("target"), port("published")
    out = []
    if published is not None and published != target:
        out.append(Draft("port", p_line, "compose", service, port=published, protocol=pr))
    if target is not None:
        out.append(Draft("port", t_line, "compose", service, port=target, protocol=pr))
    return out


def compose_drafts(root: F.Node) -> list[Draft]:
    out: list[Draft] = []
    services = _child(root, "services")
    if services is None or services.kind != "m":
        return out
    for name, svc, _ in F.bounded_entries(services):
        if svc.kind != "m":
            continue
        service = _service_name(name)
        for key in ("ports", "expose"):
            lst = _child(svc, key)
            for el in (lst.items if lst is not None and lst.kind == "l" else []):
                if el.kind == "s":
                    out += [Draft("port", el.line, "compose", service, port=p, protocol=pr) for p, pr in parse_port_spec(el.text)]
                elif el.kind == "m" and key == "ports":
                    out += _long_port(el, service)
        has_entrypoint = _child(svc, "entrypoint") is not None
        for key in ("entrypoint", "command"):
            node = _child(svc, key)
            if node is None:
                continue
            elements = _split_shell(node.text) if node.kind == "s" else _scalar_list(node)
            argv = _argv(elements or [])
            if argv:
                out.append(Draft("command", node.line, "compose", service, argv=argv,
                                 program_known=not (key == "command" and has_entrypoint)))
        build = _child(svc, "build")
        ctx = build if build is not None and build.kind == "s" else (_child(build, "context") if build is not None else None)
        if ctx is not None and ctx.kind == "s" and _safe_context(ctx.text):
            out.append(Draft("mount", ctx.line, "compose", service, mount_path=ctx.text))
    return out


def _safe_context(text: str) -> bool:
    """A build context is recorded only when it is a plain *relative* path: an absolute one (`/var/run`) is withheld, a relative one
    may leave its repo (resolution is the linker's). A git URL with credentials or an interpolated value never matches, and every
    segment must pass the token screen."""
    return (0 < len(text) <= 1024 and not text.startswith("/") and _SAFE_CONTEXT.fullmatch(text) is not None
            and not any(bundle.screen_fails(seg) for seg in text.split("/") if seg))


# --- .mcp.json and Kiro agents ---------------------------------------------------------------------------------

def mcp_drafts(root: F.Node) -> list[Draft]:
    """The launch chain's first link: each server's ``command`` + ``args`` (the server name is the ``service``)."""
    out: list[Draft] = []
    servers = _child(root, "mcpServers")
    if servers is None or servers.kind != "m":
        return out
    for name, srv, _ in F.bounded_entries(servers):
        command = _child(srv, "command")
        if command is None or command.kind != "s":
            continue
        args = _child(srv, "args")
        tail = [] if args is None else _scalar_list(args)
        argv = _argv([command.text, *tail]) if tail is not None else None
        if argv:
            out.append(Draft("command", command.line, "explicit", _service_name(name), argv=argv))
    return out
