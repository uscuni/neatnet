"""Compare neatnet-rs (Rust) output against Python neatnet on Apalachicola."""

import geopandas as gpd
import neatnet
import neatnet_rs
import numpy as np
import pytest
from shapely import wkt
from shapely.geometry import LineString


DATA_PATH = "../../crates/neatnet-core/tests/data/apalachicola_input.wkt"


@pytest.fixture
def input_wkts():
    with open(DATA_PATH) as f:
        return [line.strip() for line in f if line.strip()]


@pytest.fixture
def input_gdf(input_wkts):
    geoms = [wkt.loads(w) for w in input_wkts]
    return gpd.GeoDataFrame(geometry=geoms, crs="EPSG:3857")


def test_rust_neatify_runs(input_wkts):
    """Rust neatify produces output without errors."""
    result = neatnet_rs.neatify(input_wkts, consolidation_tolerance=10.0)
    assert len(result["geometries"]) > 0
    assert len(result["geometries"]) == len(result["statuses"])
    assert all(s in ("original", "changed", "new") for s in result["statuses"])


def test_rust_vs_python_edge_count(input_wkts, input_gdf):
    """Rust edge count is within 10% of Python."""
    rust_result = neatnet_rs.neatify(input_wkts, consolidation_tolerance=10.0)
    rust_count = len(rust_result["geometries"])

    py_result = neatnet.neatify(input_gdf)
    py_count = len(py_result)

    ratio = rust_count / py_count
    print(f"Rust: {rust_count} edges, Python: {py_count} edges, ratio: {ratio:.3f}")
    assert 0.90 <= ratio <= 1.10, (
        f"Edge count ratio {ratio:.3f} outside 10% tolerance"
    )


def test_rust_vs_python_total_length(input_wkts, input_gdf):
    """Rust total line length is within 10% of Python."""
    rust_result = neatnet_rs.neatify(input_wkts, consolidation_tolerance=10.0)
    rust_geoms = [wkt.loads(w) for w in rust_result["geometries"]]
    rust_length = sum(g.length for g in rust_geoms)

    py_result = neatnet.neatify(input_gdf)
    py_length = py_result.geometry.length.sum()

    ratio = rust_length / py_length
    print(f"Rust: {rust_length:.0f}, Python: {py_length:.0f}, ratio: {ratio:.3f}")
    assert 0.90 <= ratio <= 1.10, (
        f"Total length ratio {ratio:.3f} outside 10% tolerance"
    )


def test_rust_coins_basic():
    """Rust COINS produces correct groupings on clean data."""
    # A simple T-junction: two collinear lines and one perpendicular
    wkts = [
        "LINESTRING (0 0, 100 0)",
        "LINESTRING (100 0, 200 0)",
        "LINESTRING (100 0, 100 100)",
    ]
    rust = neatnet_rs.coins(wkts, angle_threshold=120.0)
    # First two should be grouped together (collinear), third separate
    assert rust["group"][0] == rust["group"][1], "Collinear lines should share a group"
    assert rust["group"][2] != rust["group"][0], "Perpendicular line should be separate"

    from momepy import COINS

    geoms = [wkt.loads(w) for w in wkts]
    py_coins = COINS(gpd.GeoDataFrame(geometry=geoms), angle_threshold=120)
    py_groups = py_coins.stroke_attribute()
    assert py_groups.iloc[0] == py_groups.iloc[1]
    assert py_groups.iloc[2] != py_groups.iloc[0]

    # Both should produce 2 strokes
    assert len(set(rust["group"])) == 2
    assert py_coins.stroke_gdf().shape[0] == 2


def test_rust_voronoi_skeleton():
    """Rust voronoi_skeleton produces valid output for parallel lines."""
    lines = [
        "LINESTRING (0 0, 0 100)",
        "LINESTRING (10 0, 10 100)",
    ]
    poly = "POLYGON ((-5 -5, 15 -5, 15 105, -5 105, -5 -5))"
    result = neatnet_rs.voronoi_skeleton(
        lines, poly, max_segment_length=1.0, clip_limit=2.0
    )
    assert len(result["edgelines"]) > 0
    # Skeleton should be roughly at x=5
    for ewkt in result["edgelines"]:
        geom = wkt.loads(ewkt)
        if isinstance(geom, LineString):
            xs = [c[0] for c in geom.coords]
            assert all(3.0 <= x <= 7.0 for x in xs), (
                f"Skeleton x coords {xs} not near center"
            )
