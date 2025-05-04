import geopandas

import neatnet


def test_fetch_milton_keynes():
    gdf = neatnet.utils.fetch_milton_keynes()

    assert isinstance(gdf, geopandas.GeoDataFrame)
    assert gdf.crs == 27700
    assert gdf.geom_type.unique().squeeze() == "LineString"
