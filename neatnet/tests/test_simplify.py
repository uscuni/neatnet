import pathlib

import geopandas
import momepy
import numpy
import pytest
import shapely
from pandas.testing import assert_frame_equal, assert_series_equal

import neatnet

test_data = pathlib.Path("neatnet", "tests", "data")
full_fua_data = pathlib.Path("data")

ci_artifacts = pathlib.Path("ci_artifacts")


AC = "apalachicola"
AC_STREETS = geopandas.read_parquet(test_data / f"{AC}_original.parquet")
AC_EXCLUSION_MASK = geopandas.GeoSeries(
    [
        shapely.Polygon(
            (
                (-9461361.807208396, 3469029.2708674935),
                (-9461009.046874022, 3469029.2708674935),
                (-9461009.046874022, 3469240.1785251377),
                (-9461361.807208396, 3469240.1785251377),
                (-9461361.807208396, 3469029.2708674935),
            )
        ),
        shapely.Polygon(
            (
                (-9461429.266819818, 3469157.7482423405),
                (-9461361.807208396, 3469157.7482423405),
                (-9461361.807208396, 3469240.1785251377),
                (-9461429.266819818, 3469240.1785251377),
                (-9461429.266819818, 3469157.7482423405),
            )
        ),
    ],
    crs=AC_STREETS.crs,
)


@pytest.mark.parametrize(
    "scenario,tol,known_length",
    [
        ("standard", 1.5, 64566.0),
        ("exclusion_mask", 1.05, 65765.0),
    ],
)
def test_neatify_small(scenario, tol, known_length):
    original = AC_STREETS.copy()

    known = geopandas.read_parquet(test_data / f"{AC}_simplified_{scenario}.parquet")
    exclusion_mask = AC_EXCLUSION_MASK.copy() if scenario == "exclusion_mask" else None

    observed = neatnet.neatify(original, exclusion_mask=exclusion_mask)
    observed_length = observed.geometry.length.sum()

    # storing GH artifacts
    artifact_dir = ci_artifacts / AC
    artifact_dir.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(artifact_dir / f"simplified_{scenario}.parquet")
    pytest.difference_plot(AC, artifact_dir, known, observed)

    assert pytest.approx(observed_length, rel=0.0001) == known_length
    assert observed.index.dtype == numpy.dtype("int64")

    assert observed.shape == known.shape
    assert_series_equal(known["_status"], observed["_status"])
    assert_frame_equal(
        known.drop(columns=["_status", "geometry"]),
        observed.drop(columns=["_status", "geometry"]),
    )

    pytest.geom_test(known, observed, tolerance=tol, aoi=f"{AC}_{scenario}")


@pytest.mark.parametrize(
    "aoi,tol,known_length",
    [
        ("aleppo_1133", 0.2, 4_361_625),
        ("auckland_869", 0.3, 1_268_048),
        ("bucaramanga_4617", 0.2, 1_681_011),
        ("douala_809", 0.1, 2_961_364),
        ("liege_1656", 0.3, 2_350_782),
        ("slc_4881", 0.3, 1_762_456),
    ],
)
def test_neatify_full_fua(aoi, tol, known_length):
    known = geopandas.read_parquet(full_fua_data / aoi / "simplified.parquet")
    observed = neatnet.neatify(
        geopandas.read_parquet(full_fua_data / aoi / "original.parquet")
    )
    observed_length = observed.geometry.length.sum()
    assert "highway" in observed.columns

    # storing GH artifacts
    artifact_dir = ci_artifacts / aoi
    artifact_dir.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(artifact_dir / "simplified.parquet")
    pytest.difference_plot(aoi, artifact_dir, known, observed)

    assert pytest.approx(observed_length, rel=0.0001) == known_length
    assert observed.index.dtype == numpy.dtype("int64")

    if pytest.ubuntu and pytest.env_type != "oldest":
        assert_series_equal(known["_status"], observed["_status"])
        assert_frame_equal(
            known.drop(columns=["_status", "geometry"]),
            observed.drop(columns=["_status", "geometry"]),
        )
        pytest.geom_test(known, observed, tolerance=tol, aoi=aoi)


# def test_already_simplified():
#     roads = geopandas.GeoDataFrame(
#         geometry=geopandas.GeoSeries(
#             [shapely.box(100, 100, 210, 110), shapely.box(210, 110, 220, 150)]
#         )
#         .map(shapely.get_coordinates)
#         .explode()
#         .pipe(lambda series: shapely.linestrings(list(zip(series[:-1], series[1:]))))
#     )


@pytest.mark.wuhan
def test_neatify_wuhan(aoi="wuhan_8989", tol=0.3, known_length=4_702_861):
    known = geopandas.read_parquet(full_fua_data / aoi / "simplified.parquet")
    observed = neatnet.neatify(
        geopandas.read_parquet(full_fua_data / aoi / "original.parquet")
    )
    observed_length = observed.geometry.length.sum()
    assert "highway" in observed.columns

    # storing GH artifacts
    artifact_dir = ci_artifacts / aoi
    artifact_dir.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(artifact_dir / "simplified.parquet")
    pytest.difference_plot(aoi, artifact_dir, known, observed)

    assert pytest.approx(observed_length, rel=0.0001) == known_length
    assert observed.index.dtype == numpy.dtype("int64")

    if pytest.ubuntu and pytest.env_type != "oldest":
        assert_series_equal(known["_status"], observed["_status"])
        assert_frame_equal(
            known.drop(columns=["_status", "geometry"]),
            observed.drop(columns=["_status", "geometry"]),
        )
        pytest.geom_test(known, observed, tolerance=tol, aoi=aoi)


def test_neatify_fallback():
    streets = geopandas.read_file(momepy.datasets.get_path("bubenec"), layer="streets")
    with pytest.warns(UserWarning, match="No threshold for artifact"):
        simple = neatnet.neatify(streets)
        # only topology is fixed
        assert simple.shape == (31, 2)


class TestCheckCRS:
    def test_projected_street_matching_mask(self):
        assert neatnet.simplify._check_input_crs(AC_STREETS, AC_EXCLUSION_MASK) is None

    def test_projected_street_no_mask(self):
        assert neatnet.simplify._check_input_crs(AC_STREETS, None) is None

    def test_projected_street_mismatch_mask(self):
        with pytest.raises(
            ValueError,
            match=(
                "The input `streets` and `exclusion_mask` data are in "
                "different coordinate reference systems. Reproject and rerun."
            ),
        ):
            neatnet.simplify._check_input_crs(
                AC_STREETS, AC_EXCLUSION_MASK.to_crs(4326)
            )

    def test_no_crs_street_no_mask(self):
        with pytest.warns(
            UserWarning,
            match=(
                "The input `streets` data does not have an assigned "
                "coordinate reference system. Assuming a projected CRS in meters."
            ),
        ):
            neatnet.simplify._check_input_crs(
                AC_STREETS.set_crs(None, allow_override=True), None
            )

    def test_projected_street_feet(self):
        with pytest.warns(
            UserWarning,
            match=(
                "The input `streets` data coordinate reference system is projected "
                "but not in meters. All `neatnet` defaults assume meters. "
                "Either reproject and rerun or proceed with caution."
            ),
        ):
            neatnet.simplify._check_input_crs(AC_STREETS.to_crs(6441), None)

    def test_geographic_street(self):
        with pytest.raises(
            ValueError,
            match=(
                "The input `streets` data are not in a projected "
                "coordinate reference system. Reproject and rerun."
            ),
        ):
            neatnet.simplify._check_input_crs(AC_STREETS.to_crs(4326), None)
