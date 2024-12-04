import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import simplify
from .artifacts import get_artifacts
from .nodes import (
    consolidate_nodes,
    fix_topology,
    induce_nodes,
    remove_false_nodes,
    split,
)
from .simplify import simplify_loop, simplify_network

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("neatnet")
