"""Plugin loading and dispatch."""
import importlib

from fx_svc.plugins.audit import AuditPlugin

REGISTRY: dict = {}


def load_plugin(name):
    return importlib.import_module(name)


def load_legacy(name):
    return __import__("plugins." + name)


def load_audit():
    return importlib.import_module("plugins.audit")


def dispatch(kind, evt):
    return REGISTRY.get(kind).run(evt)


def fixed_dispatch(evt):
    return AuditPlugin().run(evt)
