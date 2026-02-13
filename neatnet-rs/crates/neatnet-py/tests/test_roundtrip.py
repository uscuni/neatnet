"""Compare neatnet-rs (Rust) output against Python neatnet on Apalachicola."""

import geopandas as gpd
import neatnet
import neatnet_rs
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


# -- WKT interface tests --


def test_wkt_neatify_runs(input_wkts):
    """WKT neatify produces output without errors."""
    result = neatnet_rs.neatify_wkt(input_wkts, consolidation_tolerance=10.0)
    assert len(result["geometries"]) > 0
    assert len(result["geometries"]) == len(result["statuses"])
    assert all(s in ("original", "changed", "new") for s in result["statuses"])


def test_wkt_vs_python_edge_count(input_wkts, input_gdf):
    """WKT Rust edge count is within 10% of Python."""
    rust_result = neatnet_rs.neatify_wkt(input_wkts, consolidation_tolerance=10.0)
    rust_count = len(rust_result["geometries"])

    py_result = neatnet.neatify(input_gdf)
    py_count = len(py_result)

    ratio = rust_count / py_count
    print(f"Rust WKT: {rust_count} edges, Python: {py_count} edges, ratio: {ratio:.3f}")
    assert 0.90 <= ratio <= 1.10, (
        f"Edge count ratio {ratio:.3f} outside 10% tolerance"
    )


def test_wkt_vs_python_total_length(input_wkts, input_gdf):
    """WKT Rust total line length is within 10% of Python."""
    rust_result = neatnet_rs.neatify_wkt(input_wkts, consolidation_tolerance=10.0)
    rust_geoms = [wkt.loads(w) for w in rust_result["geometries"]]
    rust_length = sum(g.length for g in rust_geoms)

    py_result = neatnet.neatify(input_gdf)
    py_length = py_result.geometry.length.sum()

    ratio = rust_length / py_length
    print(f"Rust: {rust_length:.0f}, Python: {py_length:.0f}, ratio: {ratio:.3f}")
    assert 0.90 <= ratio <= 1.10, (
        f"Total length ratio {ratio:.3f} outside 10% tolerance"
    )


# -- GeoArrow interface tests --


def test_geoarrow_neatify_runs(input_gdf):
    """GeoArrow neatify produces output without errors."""
    ga = input_gdf.geometry.to_arrow(geometry_encoding="geoarrow")
    result_geom, result_status = neatnet_rs.neatify(ga, consolidation_tolerance=10.0)
    status_list = result_status.to_pylist()
    assert len(status_list) > 0
    assert all(s in ("original", "changed", "new") for s in status_list)


def test_geoarrow_vs_python_edge_count(input_gdf):
    """GeoArrow Rust edge count is within 10% of Python."""
    ga = input_gdf.geometry.to_arrow(geometry_encoding="geoarrow")
    result_geom, result_status = neatnet_rs.neatify(ga, consolidation_tolerance=10.0)
    rust_count = len(result_status.to_pylist())

    py_result = neatnet.neatify(input_gdf)
    py_count = len(py_result)

    ratio = rust_count / py_count
    print(f"Rust GeoArrow: {rust_count} edges, Python: {py_count} edges, ratio: {ratio:.3f}")
    assert 0.90 <= ratio <= 1.10, (
        f"Edge count ratio {ratio:.3f} outside 10% tolerance"
    )


def test_geoarrow_coins():
    """GeoArrow COINS produces correct groupings."""
    geoms = [
        LineString([(0, 0), (100, 0)]),
        LineString([(100, 0), (200, 0)]),
        LineString([(100, 0), (100, 100)]),
    ]
    gdf = gpd.GeoDataFrame(geometry=geoms)
    ga = gdf.geometry.to_arrow(geometry_encoding="geoarrow")

    result = neatnet_rs.coins(ga, angle_threshold=120.0)
    assert result["group"][0] == result["group"][1], "Collinear lines should share a group"
    assert result["group"][2] != result["group"][0], "Perpendicular line should be separate"
    assert len(set(result["group"])) == 2


# -- WKT-only function tests --


def test_coins_wkt_basic():
    """WKT COINS produces correct groupings on clean data."""
    wkts = [
        "LINESTRING (0 0, 100 0)",
        "LINESTRING (100 0, 200 0)",
        "LINESTRING (100 0, 100 100)",
    ]
    result = neatnet_rs.coins_wkt(wkts, angle_threshold=120.0)
    assert result["group"][0] == result["group"][1]
    assert result["group"][2] != result["group"][0]
    assert len(set(result["group"])) == 2

    from momepy import COINS

    geoms = [wkt.loads(w) for w in wkts]
    py_coins = COINS(gpd.GeoDataFrame(geometry=geoms), angle_threshold=120)
    assert py_coins.stroke_gdf().shape[0] == 2


def test_voronoi_skeleton_wkt():
    """WKT voronoi_skeleton produces valid output for parallel lines."""
    lines = [
        "LINESTRING (0 0, 0 100)",
        "LINESTRING (10 0, 10 100)",
    ]
    poly = "POLYGON ((-5 -5, 15 -5, 15 105, -5 105, -5 -5))"
    result = neatnet_rs.voronoi_skeleton_wkt(
        lines, poly, max_segment_length=1.0, clip_limit=2.0
    )
    assert len(result["edgelines"]) > 0
    for ewkt in result["edgelines"]:
        geom = wkt.loads(ewkt)
        if isinstance(geom, LineString):
            xs = [c[0] for c in geom.coords]
            assert all(3.0 <= x <= 7.0 for x in xs), (
                f"Skeleton x coords {xs} not near center"
            )
