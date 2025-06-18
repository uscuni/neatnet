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
        collection1 = geopandas.GeoDataFrame(geometry=collection1)
        collection1.geometry = collection1.geometry.normalize()  # type: ignore[attr-defined]

    if not is_geopandas(collection2):
        collection2 = geopandas.GeoDataFrame(geometry=collection2)
        collection2.geometry = collection2.geometry.normalize()  # type: ignore[attr-defined]

    geoms1 = collection1.geometry  # type: ignore[valid-type,attr-defined]
    geoms2 = collection2.geometry  # type: ignore[valid-type,attr-defined]

    try:
        assert shapely.equals_exact(geoms1, geoms2, tolerance=tolerance).all()
    except AssertionError:
        unexpected_bad = {}

        do_checks = []
        equal_geoms = []
        equal_topos = []

        for row in collection1.itertuples():  # type: ignore[valid-type,attr-defined]
            ix = row.Index

            # skip if known bad case
            do_check = ix not in KNOWN_BAD_GEOMS[aoi]  # type: ignore[index]
            do_checks.append(do_check)  ################################################

            # determine geometry equivalence
            g1 = collection1.loc[ix].geometry  # type: ignore[valid-type,attr-defined]
            g2 = collection2.loc[ix].geometry  # type: ignore[valid-type,attr-defined]
            equal_geom = shapely.equals_exact(g1, g2, tolerance=tolerance)
            equal_geoms.append(equal_geom)  ############################################

            # determine topological equivalence
            g1_topo = len(collection1.sindex.query(g1, predicate="touches"))  # type: ignore[valid-type,attr-defined]
            g2_topo = len(collection2.sindex.query(g2, predicate="touches"))  # type: ignore[valid-type,attr-defined]
            equal_topo = g1_topo == g2_topo
            equal_topos.append(equal_topo)  ############################################

            if do_check and not equal_geom and not equal_topo:
                # constituent coordinates per geometry
                g1_n_coords = shapely.get_coordinates(g1).shape[0]
                g2_n_coords = shapely.get_coordinates(g2).shape[0]

                # length per geometry
                g1_len = g1.length
                g2_len = g2.length

                # original index per geometry
                g1_curr_ix = collection2.loc[ix, "non_norm_ix"]  # type: ignore[valid-type,attr-defined]
                g2_prop_ix = collection2.loc[ix, "non_norm_ix"]  # type: ignore[valid-type,attr-defined]

                unexpected_bad[ix] = {
                    "n_coords": {"g1": g1_n_coords, "g2": g2_n_coords},
                    "length": {"g1": g1_len, "g2": g2_len},
                    "n_neigbors": {"g1": g1_topo, "g2": g2_topo},
                    "non_norm_ix": {"g1": g1_curr_ix, "g2": g2_prop_ix},
                }

        print(f"\n{aoi=}\n\n")
        print(f"\n{sum(do_checks)=}\n\n")
        print(f"\n{sum(equal_geoms)=}\n\n")
        print(f"\n{sum(equal_topos)=}\n\n")
        print(f"\n{unexpected_bad}\n\n")

        raise RuntimeError("stop and print") from None

        if unexpected_bad:
            # record normalized index from known and subset
            known_ix = list(unexpected_bad.keys())
            curr_compare = collection1.loc[known_ix].copy()  # type: ignore[valid-type,attr-defined]
            prop_compare = collection2.loc[known_ix].copy()  # type: ignore[valid-type,attr-defined]

            # record original, non-normalized index from known & observed
            curr_ixs_norm = [v["g1"]["non_norm_ix"] for k, v in unexpected_bad.items()]
            curr_compare["non_norm_ix"] = curr_ixs_norm

            prop_ixs_norm = [v["g2"]["non_norm_ix"] for k, v in unexpected_bad.items()]
            prop_compare["non_norm_ix"] = prop_ixs_norm

            # record $n$ neighbors for each known & observed
            curr_neighs = [v["g1"]["n_neigbors"] for k, v in unexpected_bad.items()]
            curr_compare["n_neigbors"] = curr_neighs

            prop_neighs = [v["g2"]["n_neigbors"] for k, v in unexpected_bad.items()]
            prop_compare["n_neigbors"] = prop_neighs

            # curate
            curr_compare.to_parquet(
                save_dir / "known_to_compare_simplified_{scenario}.parquet"
            )
            prop_compare.to_parquet(
                save_dir / "observed_to_compare_simplified_{scenario}.parquet"
            )

            raise AssertionError(
                f"Problem in '{aoi}' – check locs:\n{unexpected_bad}"
            ) from None

    return True


def norm_sort(gdf: geopandas.GeoDataFrame) -> geopandas.GeoDataFrame:
    """Sort GeoDataFrame by normalized geometry."""
    gdf.geometry = gdf.normalize()
    gdf = gdf.sort_values(by="geometry").reset_index(names="non_norm_ix")
    return gdf


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
    matplotlib.pyplot.savefig(writedir / f"{aoi}.png", dpi=400, bbox_inches="tight")


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
    pytest.norm_sort = norm_sort
    pytest.difference_plot = difference_plot
