"""Joern facts -> bundle records (C-1001 P1-05): the production port of the vertical slice's ``normalize.py``.

Layout: ``core`` (language-neutral: ``Source``, ``Scope``, ``Builder``, name/path rules, ``Resolver``),
``python`` (Python frontend: scope tree, spans, imports). The TS/JS frontend, blame and bundle assembly are the
second P1-05 dispatch. Runs on the host, outside the sandbox.
"""
