import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import simplify
from .artifacts import FaceArtifacts, get_artifacts
from .gaps import close_gaps, extend_lines
from .nodes import (
    consolidate_nodes,
    fix_topology,
    induce_nodes,
    remove_interstitial_nodes,
    split,
)
from .simplify import (
    neatify,
    neatify_clusters,
    neatify_loop,
    neatify_pairs,
    neatify_singletons,
)

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("neatnet")
