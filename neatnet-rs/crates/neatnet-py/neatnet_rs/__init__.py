"""neatnet-rs: Rust-accelerated street network simplification."""

from neatnet_rs._neatnet_rs import (
    coins,
    coins_wkt,
    neatify,
    neatify_wkt,
    version,
    voronoi_skeleton_wkt,
)

__all__ = [
    "coins",
    "coins_wkt",
    "neatify",
    "neatify_wkt",
    "version",
    "voronoi_skeleton_wkt",
]
__version__ = version()
