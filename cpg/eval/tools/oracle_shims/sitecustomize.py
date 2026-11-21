"""Runtime-oracle Python HTTP-client shim (P1-47; ``-p1-ml.md`` §6.5 source 2/3).

Loaded by putting this directory on PYTHONPATH (``oracle_capture.py run`` does). When ORACLE_LOG is set it
records every HTTP request issued through ``http.client`` (urllib, requests/urllib3) and ``httpx`` as one JSONL
row ``{src,method,host,port,path,frame,cls}``. ``frame`` is the first stack frame whose file lies under
ORACLE_ROOT (the ``git archive`` export) and is not a test-support/shim file: ``{file, func}`` relative to the
root, i.e. function grain. No such frame -> ``frame: null`` (the scorer counts it ``unattributed``, never guessed).
``cls`` is ``testclient`` for the Starlette TestClient host ``testserver`` (reported outside the denominator),
else ``app``. The shim never changes a request and swallows its own errors.
"""
import json
import os
import sys
import threading

_LOG = os.environ.get("ORACLE_LOG")
_ROOT = os.path.realpath(os.environ.get("ORACLE_ROOT", "")) if os.environ.get("ORACLE_ROOT") else ""
_HERE = os.path.dirname(os.path.realpath(__file__))
_lock = threading.Lock()


def _frame():
    if not _ROOT:
        return None
    f = sys._getframe(2)
    while f is not None:
        p = os.path.realpath(f.f_code.co_filename)
        if p.startswith(_ROOT + os.sep) and not p.startswith(_HERE) and "/site-packages/" not in p \
                and "/.venv/" not in p and "/node_modules/" not in p:
            return {"file": os.path.relpath(p, _ROOT), "func": f.f_code.co_name}
        f = f.f_back
    return None


def _log(method, host, port, path):
    try:
        row = {"src": "py", "method": str(method).upper(), "host": host, "port": port,
               "path": path, "frame": _frame(), "cls": "testclient" if host == "testserver" else "app"}
        line = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with _lock, open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:  # noqa: BLE001 - instrumentation must never break the caller
        pass


def _install():
    import http.client as hc
    orig_put = hc.HTTPConnection.putrequest

    def putrequest(self, method, url, *a, **k):
        _log(method, self.host, self.port, url)
        return orig_put(self, method, url, *a, **k)
    hc.HTTPConnection.putrequest = putrequest
    try:
        import httpx
        from urllib.parse import urlsplit
        for cls_ in (httpx.Client, httpx.AsyncClient):
            orig = cls_.send

            if cls_ is httpx.Client:
                def send(self, request, *a, __o=orig, **k):
                    u = request.url
                    _log(request.method, u.host, u.port, u.raw_path.decode())
                    return __o(self, request, *a, **k)
            else:
                async def send(self, request, *a, __o=orig, **k):
                    u = request.url
                    _log(request.method, u.host, u.port, u.raw_path.decode())
                    return await __o(self, request, *a, **k)
            cls_.send = send
    except ImportError:
        pass


if _LOG:
    _install()
