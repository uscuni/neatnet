import pathlib

import geopandas
import numpy
import pytest
import shapely
from pandas.testing import assert_frame_equal, assert_series_equal

import neatnet

test_data = pathlib.Path("neatnet", "tests", "data")
full_fua_data = pathlib.Path("data")

ci_artifacts = pathlib.Path("ci_artifacts")


@pytest.mark.parametrize(
    "scenario,tol,known_length",
    [
        ("standard", 1.5, 64566.0),
        ("exclusion_mask", 1.05, 65765.0),
    ],
)
def test_simplify_network_small(scenario, tol, known_length):
    ac = "apalachicola"

    original = geopandas.read_parquet(test_data / f"{ac}_original.parquet")

    known = geopandas.read_parquet(test_data / f"{ac}_simplified_{scenario}.parquet")

    if scenario == "exclusion_mask":
        exclusion_mask = [
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
        ]
        exclusion_mask = geopandas.GeoSeries(exclusion_mask, crs=original.crs)
    else:
        exclusion_mask = None

    observed = neatnet.simplify_network(original, exclusion_mask=exclusion_mask)
    observed_length = observed.geometry.length.sum()

    # storing GH artifacts
    artifact_dir = ci_artifacts / ac
    artifact_dir.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(artifact_dir / f"simplified_{scenario}.parquet")

    assert pytest.approx(observed_length, rel=0.0001) == known_length
    assert observed.index.dtype == numpy.dtype("int64")

    assert observed.shape == known.shape
    assert_series_equal(known["_status"], observed["_status"])
    assert_frame_equal(
        known.drop(columns=["_status", "geometry"]),
        observed.drop(columns=["_status", "geometry"]),
    )

    pytest.geom_test(known, observed, tolerance=tol, aoi=f"{ac}_{scenario}")


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
def test_simplify_network_full_fua(aoi, tol, known_length):
    known = geopandas.read_parquet(full_fua_data / aoi / "simplified.parquet")
    observed = neatnet.simplify_network(
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


@pytest.mark.wuhan
def test_simplify_network_wuhan(aoi="wuhan_8989", tol=0.3, known_length=4_702_861):
    known = geopandas.read_parquet(full_fua_data / aoi / "simplified.parquet")
    observed = neatnet.simplify_network(
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
