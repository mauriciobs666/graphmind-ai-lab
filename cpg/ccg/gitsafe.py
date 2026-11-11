"""The one way to run ``git`` against a repository under analysis (security review of bundle spec6, m3).

The repository is hostile: its own ``.git/config`` and ``.gitattributes`` can make git run commands (``core.fsmonitor`` on
an index read, ``diff.<driver>.textconv`` on ``git blame``, ``core.hooksPath`` hooks, ``core.pager``, ``core.sshCommand``,
a ``filter.<driver>`` on checkout). ``git(repo, *args)`` therefore runs with

- no ``GIT_*`` variable of the caller (``GIT_DIR``-style redirection, ``GIT_CONFIG_COUNT`` injection, ``GIT_SSH_COMMAND``),
  no host or system configuration (``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_SYSTEM`` are ``/dev/null``) and ``HOME`` and
  ``XDG_CONFIG_HOME`` in one empty directory;
- ``-c`` overrides of every repository setting that names a command (command line beats the repository's config);
- ``GIT_ALLOW_PROTOCOL=none`` and ``GIT_NO_LAZY_FETCH=1``: a repository's own ``protocol.<name>.allow`` beats ``-c protocol.allow``,
  and a promisor remote (``ext::<command>``) plus a missing object would run its command on any read; no transport is ever wanted;
- ``--no-replace-objects`` (a ``refs/replace`` entry must not change the bytes read) and ``--literal-pathspecs``;
- only the read-only subcommands of ``READ_ONLY``; a diff reader gets ``--no-ext-diff --no-textconv`` (and ``--no-show-signature``) from
  the helper, ``cat-file --textconv/--filters`` is refused; a NUL in an argument is a ``GitError``;
- a repository that redirects or rewrites its own storage or history is refused (``UnsafeRepository``): a ``.git`` that is a file, ``commondir``,
  ``objects/info/alternates`` (objects read from anywhere on the host) and ``info/grafts`` (a forged history; ``--no-replace-objects`` does not
  stop it). A shallow clone is not refused here: ``rev-parse --is-shallow-repository`` reports it and ``blame.require_full_history`` refuses it;
- no shell, a closed stdin, a timeout and a cap on stdout.

In-tree ``.gitattributes`` is not neutralised by ``--attr-source``: the empty tree's id depends on the object format (SHA-1 or SHA-256), and
the attributes only matter to a diff or filter reader, which the helper already forces to ``--no-ext-diff --no-textconv``/refuses.

``safe.directory`` is deliberately not set: ``*`` would switch off git's ownership check (a repository owned by another
user is refused, which the callers treat as "not listable"). A caller that reads a diff passes ``--no-ext-diff
--no-textconv`` itself (``blame`` is a diff reader and takes ``--no-textconv`` in ``extractors/joern/normalize/blame.py``);
a caller passes paths after ``--`` (``paths=`` does it).
"""
from __future__ import annotations

import atexit
import os
import resource
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT = 300                       # seconds
DEFAULT_MAX_OUTPUT = 256 * 1024 * 1024      # bytes of stdout

# Every repository setting that names a command or a network, forced off. (`core.editor`, `credential.helper` and the
# `uploadpack`/`receivepack` family need an interactive or network command no read-only call reaches.)
HARDENING_CONFIG = (
    "core.fsmonitor=false", "core.hooksPath=/dev/null", "core.pager=cat", "protocol.allow=never", "core.sshCommand=false",
    "core.gitProxy=false", "core.askPass=false", "core.attributesFile=/dev/null",
    "log.showSignature=false", "gpg.program=false", "gpg.openpgp.program=false", "gpg.x509.program=false", "gpg.ssh.program=false",
)


class GitError(subprocess.SubprocessError):
    """A git call the helper stopped."""


class GitTimeout(GitError):
    """The call ran past its timeout."""


class GitOutputTooLarge(GitError):
    """The call wrote more than its output cap."""


class UnsafeRepository(GitError):
    """The repository redirects or rewrites its own storage or history, or the call is one the helper does not run. Value-free: the
    message names the kind of problem and a path relative to the git directory, never a repository-chosen value."""


#: The subcommands the helper runs; any other (``checkout``, ``status``, ``fetch``, ``config``...) is refused.
READ_ONLY = frozenset({"ls-files", "ls-tree", "rev-parse", "cat-file", "rev-list", "log", "show", "diff", "blame", "whatchanged",
                       "format-patch", "show-ref", "for-each-ref", "merge-base"})
DIFF_READERS = frozenset({"diff", "show", "log", "whatchanged", "format-patch", "blame"})
SIGNATURE_READERS = frozenset({"show", "log", "whatchanged"})
_UNSAFE_CAT_FILE = ("--textconv", "--filters", "--path")
#: Long options a caller may not pass (nor any unambiguous abbreviation of them: git accepts ``--ext-d``): they re-enable a repository
#: driver, redirect git to a directory, or make git read or write a caller-named path. Short ones: ``-c``, ``-o``, ``-O...``.
_REFUSED_LONG = ("ext-diff", "textconv", "output", "output-directory", "git-dir", "work-tree", "exec-path", "config-env", "attr-source",
                 "open-files-in-pager", "no-index", "namespace", "super-prefix")
_REDIRECTS = ("commondir", "objects/info/alternates", "info/grafts")


_empty_home: str | None = None


def _home() -> str:
    global _empty_home
    if _empty_home is None or not os.path.isdir(_empty_home):
        _empty_home = tempfile.mkdtemp(prefix="gitsafe-home-")
        atexit.register(shutil.rmtree, _empty_home, True)
    return _empty_home


def environment() -> dict[str, str]:
    """The environment of every git call (see the module docstring)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    home = _home()
    env.update(HOME=home, XDG_CONFIG_HOME=home, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_CONFIG_NOSYSTEM="1",
               GIT_ALLOW_PROTOCOL="none", GIT_NO_LAZY_FETCH="1")
    return env


def command(repo, *args: str) -> list[str]:
    """The ``git`` argv for ``args`` run in ``repo`` (also what a streaming caller passes to ``Popen`` with ``environment()``)."""
    argv = ["git", "--no-replace-objects", "--literal-pathspecs"]
    for setting in HARDENING_CONFIG:
        argv += ["-c", setting]
    return [*argv, "-C", str(repo), *args]


def _git_dir(repo) -> Path | None:
    """The git directory git would discover from ``repo`` (a work tree's ``.git``, found upward as git does), or ``repo`` itself when it is a
    bare repository; ``None`` when neither (git then fails on its own). A ``.git`` that is not a directory is refused."""
    here = Path(repo).resolve()
    for d in (here, *here.parents):
        dot = d / ".git"
        if dot.is_symlink() or dot.is_file():
            raise UnsafeRepository(".git is not a directory (a gitdir redirect)")
        if dot.is_dir():
            return dot
    if (here / "HEAD").is_file() and (here / "objects").is_dir():
        return here
    return None


def check_repository(repo) -> None:
    """``UnsafeRepository`` for a repository that redirects or rewrites its own storage or history (see the module docstring)."""
    gd = _git_dir(repo)
    for rel in _REDIRECTS if gd else ():
        if os.path.lexists(gd / rel):
            raise UnsafeRepository(f"the repository has {rel}")


def _checked(args: tuple[str, ...], paths) -> list[str]:
    """``args`` with the diff-reader flags added; ``UnsafeRepository``/``GitError`` for a call the helper does not run."""
    everything = [*args, *map(str, paths)]
    if any("\0" in a for a in everything):
        raise GitError("a NUL byte in a git argument")
    sub = args[0] if args else ""
    if sub not in READ_ONLY:
        raise UnsafeRepository(f"git subcommand {sub!r} is not one the helper runs")
    if sub == "cat-file" and any(a.split("=")[0] in _UNSAFE_CAT_FILE for a in args[1:]):
        raise UnsafeRepository("cat-file with a textconv or filter is not run")
    own = list(args[1:])
    end = own.index("--") if "--" in own else len(own)
    for a in own[:end]:
        name = a[2:].split("=", 1)[0]
        if (a.startswith("--") and name and any(r.startswith(name) for r in _REFUSED_LONG)) or a in ("-c", "-o") or a.startswith("-O"):
            raise UnsafeRepository("an option that re-enables a repository driver or names a path is not run")
    extra = []
    if sub in DIFF_READERS:
        extra += ["--no-ext-diff", "--no-textconv"]
    if sub in SIGNATURE_READERS:
        extra += ["--no-show-signature"]
    return [sub, *own[:end], *extra, *own[end:]]                    # last among the options: a later flag beats an earlier one


def checked_command(repo, *args: str) -> list[str]:
    """``command()`` for a streaming caller, after the helper's own checks: the repository (``check_repository``) and the subcommand and
    options (``READ_ONLY`` and the refused options). Run it with ``environment()``."""
    check_repository(repo)
    return command(repo, *_checked(args, ()))


def git(repo, *args: str, paths=(), timeout: float | None = None, max_output: int | None = None) -> bytes:
    """Stdout of ``git`` run in ``repo``. ``paths`` follow a ``--``. ``CalledProcessError`` on a git failure, ``GitTimeout``
    past ``timeout``, ``GitOutputTooLarge`` past ``max_output`` bytes of stdout."""
    timeout = DEFAULT_TIMEOUT if timeout is None else timeout
    cap = DEFAULT_MAX_OUTPUT if max_output is None else max_output
    argv = command(repo, *_checked(args, paths)) + (["--", *map(str, paths)] if paths else [])
    check_repository(repo)

    def limit() -> None:                                           # a write past the cap kills git (SIGXFSZ) before the disk fills
        resource.setrlimit(resource.RLIMIT_FSIZE, (cap + 1, cap + 1))

    with tempfile.TemporaryFile() as out:
        try:
            done = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.PIPE, check=False,
                                  timeout=timeout, env=environment(), shell=False, preexec_fn=limit)
        except subprocess.TimeoutExpired as e:
            raise GitTimeout(f"git {args[0] if args else ''} timed out after {timeout} s") from e
        size = out.seek(0, os.SEEK_END)
        if size > cap:                                                 # the limit is cap + 1, so a killed writer is past the cap
            raise GitOutputTooLarge(f"git {args[0] if args else ''} wrote more than {cap} bytes")
        out.seek(0)
        data = out.read()
    if done.returncode:
        raise subprocess.CalledProcessError(done.returncode, argv, data, done.stderr)
    return data
