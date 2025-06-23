import pathlib
import platform
import warnings

import geopandas.testing
import matplotlib.pyplot
import numpy
import pandas
import pytest
import shapely

import neatnet

# set the global exception raiser for testing & debugging
# See gh#121
neatnet.simplify.DEBUGGING = False

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
    "wuhan_8989": [],
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
    save_dir: pathlib.Path = pathlib.Path(""),
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
        _per_edge_check(geoms1, geoms2, tolerance, aoi, save_dir)

    return True


def _per_edge_check(geoms1, geoms2, tolerance, aoi, save_dir):
    """Granular 1-1 comparison of known vs. observed edges. Curates offenders.

    Checkes & records:
        * number of coordinates
        * length
        * number of neighbors (touching - excluding self)

    See ``geom_test()`` for parameter descriptions.
    """

    def _n_touches(edges: geopandas.GeoSeries, edge: shapely.LineString) -> int:
        """number of neighbors (touching - excluding self)"""
        return len(edges.sindex.query(edge, predicate="touches"))

    def _prep_compare(compare_cases: dict):
        """Convert offender info into geodataframe and save."""
        known: dict = {
            "index": [],
            "n_coords": [],
            "length": [],
            "n_neigbors": [],
            "geometry": [],
        }
        observed: dict = {
            "index": [],
            "n_coords": [],
            "length": [],
            "n_neigbors": [],
            "geometry": [],
        }

        for ix, info in compare_cases.items():
            # details of the known edge
            known["index"].append(ix)
            known["n_coords"].append(info["n_coords"]["g1"])
            known["length"].append(info["length"]["g1"])
            known["n_neigbors"].append(info["n_neigbors"]["g1"])
            known["geometry"].append(info["geometry"]["g1"])

            # details of the observed edge
            observed["index"].append(ix)
            observed["n_coords"].append(info["n_coords"]["g2"])
            observed["length"].append(info["length"]["g2"])
            observed["n_neigbors"].append(info["n_neigbors"]["g2"])
            observed["geometry"].append(info["geometry"]["g2"])

        # curate
        known_gdf = geopandas.GeoDataFrame.from_dict(known, crs=geoms1.crs)
        known_gdf.to_parquet(save_dir / "known_to_compare_simplified.parquet")

        observed_gdf = geopandas.GeoDataFrame.from_dict(observed, crs=geoms2.crs)
        observed_gdf.to_parquet(save_dir / "observed_to_compare_simplified.parquet")

    unexpected_bad = {}
    do_checks = []
    equal_geoms = []
    equal_topos = []

    for ix in geoms1.index:
        # skip if known bad case
        do_check = ix not in KNOWN_BAD_GEOMS[aoi]
        do_checks.append(do_check)

        # determine geometry equivalence
        g1 = geoms1.loc[ix]
        g2 = geoms2.loc[ix]
        equal_geom = shapely.equals_exact(g1, g2, tolerance=tolerance)
        equal_geoms.append(equal_geom)

        # determine topological equivalence
        g1_topo = _n_touches(geoms1, g1)
        g2_topo = _n_touches(geoms2, g2)
        equal_topo = g1_topo == g2_topo
        equal_topos.append(equal_topo)

        if do_check and (not equal_geom or not equal_topo):
            # constituent coordinates per geometry
            g1_n_coords = shapely.get_coordinates(g1).shape[0]
            g2_n_coords = shapely.get_coordinates(g2).shape[0]

            # length per geometry
            g1_len = g1.length
            g2_len = g2.length

            unexpected_bad[ix] = {
                "n_coords": {"g1": g1_n_coords, "g2": g2_n_coords},
                "length": {"g1": g1_len, "g2": g2_len},
                "n_neigbors": {"g1": g1_topo, "g2": g2_topo},
                "geometry": {"g1": g1, "g2": g2},
            }

    if unexpected_bad:
        if save_dir:
            _prep_compare(unexpected_bad)

        n_geoms = len(do_checks)
        raise AssertionError(
            f"Problem in '{aoi}'\n\n"
            f"Total geoms:   {n_geoms}\n"
            f"Checked geoms: {sum(do_checks)}\n"
            f"!= geoms:      {n_geoms - sum(equal_geoms)}\n"
            f"!= topos:      {n_geoms - sum(equal_topos)}\n\n"
            f"Check locs:\n{unexpected_bad}"
        ) from None


def difference_plot(
    aoi: str,
    writedir: pathlib.Path,
    known: geopandas.GeoDataFrame,
    observed: geopandas.GeoDataFrame,
    diff_buff: int = 50,
):
    """Plot difference locations observed simplified in relation to known simplified."""

    crs = known.crs

    # unioned multilinestring of each - known & observed
    known = geopandas.GeoDataFrame(geometry=[known.union_all()], crs=crs)
    observed = geopandas.GeoDataFrame(geometry=[observed.union_all()], crs=crs)

    # unioned difference of k-o + o-k
    known_observed_diff = known.difference(observed)
    observed_known_diff = observed.difference(known)
    differences = geopandas.GeoDataFrame(
        geometry=[
            pandas.concat([known_observed_diff, observed_known_diff])
            .explode()
            .union_all()
        ],
        crs=crs,
    )

    # plot difference locations in relation to known
    base = known.plot(figsize=(15, 15), zorder=2, alpha=0.4, ec="k", lw=0.5)
    with warnings.catch_warnings():
        # See GL#188
        warnings.filterwarnings(
            "ignore",
            message="The GeoSeries you are attempting to plot",
            category=UserWarning,
        )
        differences.buffer(diff_buff).plot(ax=base, zorder=1, fc="r", alpha=0.6)
    base.set_title(f"known vs. observed differences - {aoi}")
    matplotlib.pyplot.savefig(writedir / f"{aoi}.png", dpi=500, bbox_inches="tight")


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
    pytest.difference_plot = difference_plot
