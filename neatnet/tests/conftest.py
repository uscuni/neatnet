import platform

import geopandas.testing
import numpy
import pandas
import pytest
import shapely

import neatnet

# set the global exception raiser for testing & debugging
# See gh#121
neatnet.simplify.DEBUGGING = True

line_collection = (  # type: ignore[valid-type, misc]
    list[shapely.LineString]
    | tuple[shapely.LineString]
    | numpy.ndarray
    | pandas.Series
    | geopandas.GeoSeries
)

geometry_collection = (
    list[shapely.GeometryCollection]
    | tuple[shapely.GeometryCollection]
    | numpy.ndarray
    | pandas.Series
    | geopandas.GeoSeries
)


####################################################
# see:
#   - gh#106
#   - gh#102
#   - gh#77
#   - gh#75
#   - gh#74
KNOWN_BAD_GEOMS = {
    "aleppo_1133": [],
    "auckland_869": [1412],
    "bucaramanga_4617": [],
    "douala_809": [],
    "liege_1656": [921],
    "slc_4881": [1144, 1146],
    "apalachicola_standard": [324],
    "apalachicola_exclusion_mask": [],
}
####################################################


def polygonize(
    collection: line_collection,  # type: ignore[valid-type]
    as_geom: bool = True,  # type: ignore[valid-type]
) -> shapely.Polygon | geopandas.GeoSeries:
    """Testing helper -- Create polygon from collection of lines."""
    if isinstance(collection, pandas.Series | geopandas.GeoSeries):
        _poly = geopandas.GeoSeries(collection).polygonize()
        if as_geom:
            return _poly.squeeze()
        else:
            return _poly
    else:
        return shapely.polygonize(collection).buffer(0)


def is_geopandas(collection: geometry_collection) -> bool:  # type: ignore[valid-type]
    return isinstance(collection, geopandas.GeoSeries | geopandas.GeoDataFrame)


def geom_test(
    collection1: geometry_collection,  # type: ignore[valid-type]
    collection2: geometry_collection,  # type: ignore[valid-type]
    tolerance: float = 1e-1,
    aoi: None | str = None,
) -> bool:
    """Testing helper -- geometry verification."""

    if not is_geopandas(collection1):
        collection1 = geopandas.GeoSeries(collection1)

    if not is_geopandas(collection2):
        collection2 = geopandas.GeoSeries(collection2)

    geoms1 = collection1.geometry.normalize()  # type: ignore[attr-defined]
    geoms2 = collection2.geometry.normalize()  # type: ignore[attr-defined]

    if aoi and aoi.startswith("apalachicola"):
        # Varied index order across OSs.
        # See [https://github.com/uscuni/neatnet/pull/104#issuecomment-2495572388]
        geoms1 = geoms1.sort_values().reset_index(drop=True)
        geoms2 = geoms2.sort_values().reset_index(drop=True)

    try:
        assert shapely.equals_exact(geoms1, geoms2, tolerance=tolerance).all()
    except AssertionError:
        unexpected_bad = {}
        for ix in geoms1.index:
            g1 = geoms1.loc[ix]
            g2 = geoms2.loc[ix]
            if (
                not shapely.equals_exact(g1, g2, tolerance=tolerance)
                and ix not in KNOWN_BAD_GEOMS[aoi]  # type: ignore[index]
            ):
                unexpected_bad[ix] = {
                    "n_coords": {
                        "g1": shapely.get_coordinates(g1).shape[0],
                        "g2": shapely.get_coordinates(g2).shape[0],
                    },
                    "length": {"g1": g1.length, "g2": g2.length},
                }
        if unexpected_bad:
            raise AssertionError(
                f"Problem in '{aoi}' – check locs: {unexpected_bad}"
            ) from None
    return True


def pytest_addoption(parser):
    """Add custom command line arguments."""

    # flag for determining CI environment
    parser.addoption(
        "--env_type",
        action="store",
        default="latest",
        help="Testing environment type label",
        type=str,
    )


def pytest_configure(config):  # noqa: ARG001
    """PyTest session attributes, methods, etc."""

    valid_env_types = ["oldest", "latest", "dev"]
    pytest.env_type = config.getoption("env_type").split("_")[-1]
    assert pytest.env_type in valid_env_types

    pytest.ubuntu = "ubuntu" in platform.version().lower()

    pytest.polygonize = polygonize
    pytest.geom_test = geom_test
