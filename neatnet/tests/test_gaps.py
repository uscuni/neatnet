import geopandas as gpd
import pytest
from shapely.geometry import LineString

import neatnet


def test_close_gaps():
    l0 = LineString([(0, 1), (0.9, 0)])
    l1 = LineString([(1, 0), (2, 1)])
    l2 = LineString([(2.1, 1), (3, 2)])
    l3 = LineString([(3.1, 2), (4, 0)])
    l4 = LineString([(4.1, 0), (5, 0)])
    l5 = LineString([(5.1, 0), (6, 0)])
    gdf = gpd.GeoDataFrame(geometry=[l0, l1, l2, l3, l4, l5])

    closed = neatnet.close_gaps(gdf, 0.25)
    assert len(closed) == len(gdf)

    merged = neatnet.remove_interstitial_nodes(closed)
    assert len(merged) == 1
    assert merged.length[0] == pytest.approx(8.4662, rel=1e-3)


def test_extend_lines():
    l1 = LineString([(1, 0), (1.9, 0)])
    l2 = LineString([(2.1, -1), (2.1, 1)])
    l3 = LineString([(2, 1.1), (3, 1.1)])
    l4 = LineString([(2.2, 0), (3, 0)])
    l5 = LineString([(1.5, 0.5), (1, 0.0)])
    l6 = LineString([(1, 0.0), (1.5, -0.5)])
    l7 = LineString([(3, -2), (2.15, -1.05)])

    edges = [l1, l2, l3, l4, l5, l6, l7]

    gdf = gpd.GeoDataFrame(range(1, len(edges) + 1), geometry=edges)

    ext1 = neatnet.extend_lines(gdf, 2)
    assert ext1.length.sum() > gdf.length.sum()
    assert ext1.length.sum() == pytest.approx(8.7124, rel=1e-3)

    target = gpd.GeoSeries([l2.centroid.buffer(3)])
    ext2 = neatnet.extend_lines(gdf, 3, target=target)
    assert ext2.length.sum() > gdf.length.sum()
    assert ext2.length.sum() == pytest.approx(33.8295, rel=1e-3)

    barrier_1 = LineString([(2, -1), (2, 1)])
    ext3 = neatnet.extend_lines(gdf, 2, barrier=gpd.GeoSeries([barrier_1]))
    assert ext3.length.sum() > gdf.length.sum()
    assert ext3.length.sum() == pytest.approx(7.6639, rel=1e-3)

    ext4 = neatnet.extend_lines(gdf, 2, extension=1)
    assert ext4.length.sum() > gdf.length.sum()
    assert ext4.length.sum() == pytest.approx(19.7124, rel=1e-3)

    gdf = gpd.GeoDataFrame([1, 2, 3, 4], geometry=[l1, l2, l3, barrier_1])

    ext5 = neatnet.extend_lines(gdf, 2)
    assert ext5.length.sum() > gdf.length.sum()
    assert ext5.length.sum() == pytest.approx(6.2, rel=1e-3)

    barrier_2 = LineString([(2.0, 1.05), (3, 1.02)])
    ext6 = neatnet.extend_lines(gdf, 2, barrier=gpd.GeoSeries([barrier_2]))
    assert ext6.length.sum() > gdf.length.sum()
    assert ext6.length.sum() == pytest.approx(6.0, rel=1e-3)
