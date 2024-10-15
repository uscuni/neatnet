import platform

import geopandas.testing
import numpy
import pandas
import pytest
import shapely

line_collection = (
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


def polygonize(
    collection: line_collection, as_geom: bool = True
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


def is_geopandas(collection: geometry_collection) -> bool:
    return isinstance(collection, geopandas.GeoSeries | geopandas.GeoDataFrame)


def geom_test(
    collection1: geometry_collection,
    collection2: geometry_collection,
    tolerance: float = 1e-1,
) -> bool:
    """Testing helper -- geometry verification."""

    if not is_geopandas(collection1):
        collection1 = geopandas.GeoSeries(collection1)

    if not is_geopandas(collection2):
        collection2 = geopandas.GeoSeries(collection2)

    assert shapely.equals_exact(
        collection1.geometry.normalize(),
        collection2.geometry.normalize(),
        tolerance=tolerance,
    ).all()


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
    pytest.env_type = config.getoption("env_type").split("-")[-1]
    assert pytest.env_type in valid_env_types

    pytest.ubuntu = "ubuntu" in platform.version().lower()

    pytest.polygonize = polygonize
    pytest.geom_test = geom_test
