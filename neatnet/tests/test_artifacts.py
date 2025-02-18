import pathlib

import geopandas
import pytest
import shapely

import neatnet


def test_get_artifacts_error():
    path = pathlib.Path("neatnet", "tests", "data", "apalachicola_original.parquet")
    with pytest.raises(  # noqa: SIM117
        ValueError,
        match=(
            "No threshold for artifact detection found. Pass explicit "
            "`threshold` or `threshold_fallback` to provide the value directly."
        ),
    ):
        with pytest.warns(
            UserWarning,
            match=(
                "Input roads could not not be polygonized. "
                "Identification of face artifacts not possible."
            ),
        ):
            neatnet.artifacts.get_artifacts(geopandas.read_parquet(path).iloc[:3])


def test_FaceArtifacts():  # noqa: N802
    pytest.importorskip("esda")
    osmnx = pytest.importorskip("osmnx")
    type_filter = (
        '["highway"~"living_street|motorway|motorway_link|pedestrian|primary'
        "|primary_link|residential|secondary|secondary_link|service|tertiary"
        '|tertiary_link|trunk|trunk_link|unclassified|service"]'
    )
    streets_graph = osmnx.graph_from_point(
        (35.7798, -78.6421),
        dist=1000,
        network_type="all_private",
        custom_filter=type_filter,
        retain_all=True,
        simplify=False,
    )
    streets_graph = osmnx.projection.project_graph(streets_graph)
    gdf = osmnx.graph_to_gdfs(
        osmnx.convert.to_undirected(streets_graph),
        nodes=False,
        edges=True,
        node_geometry=False,
        fill_edge_geometry=True,
    )
    fa = neatnet.FaceArtifacts(gdf)
    assert 6 < fa.threshold < 9
    assert isinstance(fa.face_artifacts, geopandas.GeoDataFrame)
    assert fa.face_artifacts.shape[0] > 200
    assert fa.face_artifacts.shape[1] == 2

    with pytest.warns(UserWarning, match="No threshold found"):
        neatnet.FaceArtifacts(gdf.cx[712104:713000, 3961073:3961500])

    fa_ipq = neatnet.FaceArtifacts(gdf, index="isoperimetric_quotient")
    assert 6 < fa_ipq.threshold < 9
    assert fa_ipq.threshold != fa.threshold

    fa_dia = neatnet.FaceArtifacts(gdf, index="diameter_ratio")
    assert 6 < fa_dia.threshold < 9
    assert fa_dia.threshold != fa.threshold

    fa = neatnet.FaceArtifacts(gdf, index="isoperimetric_quotient")
    assert 6 < fa.threshold < 9

    with pytest.raises(ValueError, match="'banana' is not supported"):
        neatnet.FaceArtifacts(gdf, index="banana")

    p1, p2, p3, p4 = (
        shapely.Point(1, 0),
        shapely.Point(2, 0),
        shapely.Point(3, 0),
        shapely.Point(2, 1),
    )
    inverted_t = [
        shapely.LineString((p1, p2)),
        shapely.LineString((p2, p3)),
        shapely.LineString((p2, p4)),
    ]

    with pytest.warns(
        UserWarning,
        match=(
            "Input roads could not not be polygonized. "
            "Identification of face artifacts not possible."
        ),
    ):
        neatnet.FaceArtifacts(geopandas.GeoDataFrame(geometry=inverted_t))
