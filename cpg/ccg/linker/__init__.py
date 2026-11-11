"""The linker (P1-17): ``link(facts, bundles) -> Links``. Dispatch 1: matchers and linking; ``bind.py`` (URL, env/Compose and
launch binding) plugs in through the :class:`~ccg.linker.link.Bindings` seam."""
from ccg.linker.link import Binding, Bindings, BoundEntry, Link, Links, Unmapped, Via, count_links, link

__all__ = ["Binding", "Bindings", "BoundEntry", "Link", "Links", "Unmapped", "Via", "count_links", "link"]
