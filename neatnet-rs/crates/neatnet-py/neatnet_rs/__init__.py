"""neatnet-rs: Rust-accelerated street network simplification."""

import geopandas as gpd

from neatnet_rs._neatnet_rs import (
    coins as _coins_arrow,
    coins_wkt,
    neatify as _neatify_arrow,
    neatify_wkt,
    version,
    voronoi_skeleton_wkt,
)


def neatify(streets: gpd.GeoDataFrame, **kwargs) -> gpd.GeoDataFrame:
    """Simplify a street network.

    Parameters
    ----------
    streets : GeoDataFrame
        Input street network with LineString geometry.
    **kwargs
        Forwarded to the Rust neatify function.

    Returns
    -------
    GeoDataFrame
        Simplified street network with geometry and status columns.
    """
    table = streets.to_arrow(geometry_encoding="geoarrow")
    result = _neatify_arrow(table, **kwargs)
    return gpd.GeoDataFrame.from_arrow(result)


def coins(streets: gpd.GeoDataFrame, *, angle_threshold: float = 120.0) -> gpd.GeoDataFrame:
    """Run COINS continuity analysis.

    Parameters
    ----------
    streets : GeoDataFrame
        Input street network with LineString geometry.
    angle_threshold : float, default 120.0
        Angle threshold for continuity detection.

    Returns
    -------
    GeoDataFrame
        Input data with additional COINS columns (group, is_end,
        stroke_length, stroke_count).
    """
    table = streets.to_arrow(geometry_encoding="geoarrow")
    result = _coins_arrow(table, angle_threshold=angle_threshold)
    return gpd.GeoDataFrame.from_arrow(result)


__all__ = [
    "coins",
    "coins_wkt",
    "neatify",
    "neatify_wkt",
    "version",
    "voronoi_skeleton_wkt",
]
__version__ = version()
