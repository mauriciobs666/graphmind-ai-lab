"""The linker (P1-17): ``link(facts, bundles) -> Links`` (``link.py``, ``matchers.py``) and the binding provider :class:`Binder`
(``bind.py``: URL, env/Compose and launch binding) that plugs in through the :class:`~ccg.linker.link.Bindings` seam."""
from ccg.linker.bind import Binder
from ccg.linker.link import Binding, Bindings, BoundEntry, Link, Links, Unmapped, Via, count_links, link

__all__ = ["Binder", "Binding", "Bindings", "BoundEntry", "Link", "Links", "Unmapped", "Via", "count_links", "link"]
