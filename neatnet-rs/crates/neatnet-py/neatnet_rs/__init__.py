"""neatnet-rs: Rust-accelerated street network simplification."""

from neatnet_rs._neatnet_rs import coins, neatify, version, voronoi_skeleton

__all__ = ["coins", "neatify", "version", "voronoi_skeleton"]
__version__ = version()
