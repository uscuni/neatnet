import geopandas as gpd
import pytest
from shapely.geometry import LineString

import neatnet


def test_close_gaps():
    l1 = LineString([(1, 0), (2, 1)])
    l2 = LineString([(2.1, 1), (3, 2)])
    l3 = LineString([(3.1, 2), (4, 0)])
    l4 = LineString([(4.1, 0), (5, 0)])
    l5 = LineString([(5.1, 0), (6, 0)])
    df = gpd.GeoDataFrame(geometry=[l1, l2, l3, l4, l5])

    closed = neatnet.close_gaps(df, 0.25)
    assert len(closed) == len(df)

    merged = neatnet.remove_false_nodes(closed)
    assert len(merged) == 1
    assert merged.length[0] == pytest.approx(7.0502, rel=1e-3)


def test_extend_lines():
    l1 = LineString([(1, 0), (1.9, 0)])
    l2 = LineString([(2.1, -1), (2.1, 1)])
    l3 = LineString([(2, 1.1), (3, 1.1)])
    gdf = gpd.GeoDataFrame([1, 2, 3], geometry=[l1, l2, l3])

    ext1 = neatnet.extend_lines(gdf, 2)
    assert ext1.length.sum() > gdf.length.sum()
    assert ext1.length.sum() == pytest.approx(4.2, rel=1e-3)

    target = gpd.GeoSeries([l2.centroid.buffer(3)])
    ext2 = neatnet.extend_lines(gdf, 3, target)

    assert ext2.length.sum() > gdf.length.sum()
    assert ext2.length.sum() == pytest.approx(17.3776, rel=1e-3)

    barrier = LineString([(2, -1), (2, 1)])
    ext3 = neatnet.extend_lines(gdf, 2, barrier=gpd.GeoSeries([barrier]))

    assert ext3.length.sum() > gdf.length.sum()
    assert ext3.length.sum() == pytest.approx(4, rel=1e-3)

    ext4 = neatnet.extend_lines(gdf, 2, extension=1)
    assert ext4.length.sum() > gdf.length.sum()
    assert ext4.length.sum() == pytest.approx(10.2, rel=1e-3)

    gdf = gpd.GeoDataFrame([1, 2, 3, 4], geometry=[l1, l2, l3, barrier])
    ext5 = neatnet.extend_lines(gdf, 2)
    assert ext5.length.sum() > gdf.length.sum()
    assert ext5.length.sum() == pytest.approx(6.2, rel=1e-3)
