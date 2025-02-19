import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import simplify
from .artifacts import FaceArtifacts, get_artifacts
from .gaps import close_gaps, extend_lines
from .nodes import (
    consolidate_nodes,
    fix_topology,
    induce_nodes,
    remove_false_nodes,
    split,
)
from .simplify import (
    simplify_clusters,
    simplify_loop,
    simplify_network,
    simplify_pairs,
    simplify_singletons,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("neatnet")
