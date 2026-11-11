"""Config extractor (C-1001 P1-11, F-6): ``configEntry`` and ``launchFact`` records from config files.

``extract.extract_config`` is the entry point; ``formats`` holds the per-format ``keyPath`` derivation table and the
bounded parsers; ``launch`` holds the launch-fact derivations. Every rule that decides ``bindValue``, ``default``, argv or
a hash is ``extractors.common.writer_rules``'s, never decided here.
"""
