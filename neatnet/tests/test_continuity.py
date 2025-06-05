import io

import geopandas.testing
import momepy
import pandas
import pytest
import shapely

import neatnet


@pytest.fixture
def streets() -> geopandas.GeoDataFrame:
    """Toy set of 'streets' for testing only."""
    inita = 2
    final = 8
    grid = list(range(inita, final))
    vert_points = list(zip(grid[:-1], grid[1:], strict=True))
    hori_points = [(j, i) for i, j in vert_points]
    vert_lines = [
        shapely.LineString(i)
        for i in list(zip(hori_points[:-1], vert_points[1:], strict=True))
    ]
    hori_lines = [
        shapely.LineString(i)
        for i in list(zip(vert_points[:-1], hori_points[1:], strict=True))
    ]
    return geopandas.GeoDataFrame(
        geometry=(
            vert_lines
            + hori_lines
            + [
                shapely.LineString(((4, 5), (3, 6))),
                shapely.LineString(((3, 6), (4, 4))),
                shapely.LineString(((6, 3), (5, 4))),
                shapely.LineString(((3, 6), (3, 4))),
                shapely.LineString(((5, 5), (6, 6))),
                shapely.LineString(((6, 7), (7, 7))),
                shapely.LineString(((7, 6), (7, 7))),
            ]
        )
    )


def test_continuity(streets):
    observed_continuity, observed_coins = neatnet.continuity.continuity(streets)

    assert isinstance(observed_continuity, geopandas.GeoDataFrame)
    known_continuity = (
        geopandas.GeoDataFrame(
            pandas.read_csv(
                io.StringIO(
                    "geometry	coins_group	coins_end	coins_len	coins_count\n"
                    "LINESTRING (3 2, 3 4)	0	True	4.0	2\n"
                    "LINESTRING (4 3, 4 5)	1	True	4.0	2\n"
                    "LINESTRING (5 4, 5 6)	2	True	11.414213562373096	7\n"
                    "LINESTRING (6 5, 6 7)	2	False	11.414213562373096	7\n"
                    "LINESTRING (2 3, 4 3)	1	True	4.0	2\n"
                    "LINESTRING (3 4, 5 4)	3	True	3.414213562373095	2\n"
                    "LINESTRING (4 5, 6 5)	2	False	11.414213562373096	7\n"
                    "LINESTRING (5 6, 7 6)	2	False	11.414213562373096	7\n"
                    "LINESTRING (4 5, 3 6)	2	True	11.414213562373096	7\n"
                    "LINESTRING (3 6, 4 4)	4	True	2.23606797749979	1\n"
                    "LINESTRING (6 3, 5 4)	3	True	3.414213562373095	2\n"
                    "LINESTRING (3 6, 3 4)	0	True	4.0	2\n"
                    "LINESTRING (5 5, 6 6)	5	True	1.4142135623730951	1\n"
                    "LINESTRING (6 7, 7 7)	2	False	11.414213562373096	7\n"
                    "LINESTRING (7 6, 7 7)	2	False	11.414213562373096	7\n"
                ),
                sep="\t",
            )
        )
        .pipe(lambda df: df.assign(**{"geometry": shapely.from_wkt(df["geometry"])}))
        .set_geometry("geometry")
    )
    geopandas.testing.assert_geodataframe_equal(observed_continuity, known_continuity)

    assert isinstance(observed_coins, momepy.COINS)
    assert observed_coins.already_merged
    assert observed_coins.merging_list == [
        [0, 11],
        [1, 4],
        [2, 3, 6, 7, 8, 13, 14],
        [5, 10],
        [9],
        [12],
    ]
    assert len(observed_coins.angle_pairs) == 36


def test_get_stroke_info(streets):
    known_strokes = [0, 0, 2, 1, 1, 1, 2]
    known_c_ = [0, 0, 0, 0, 0, 1, 0]
    known_e_ = [0, 0, 1, 0, 0, 0, 1]
    known_s_ = [0, 0, 1, 1, 1, 0, 1]

    observed = neatnet.continuity.get_stroke_info(
        neatnet.artifacts.get_artifacts(streets, threshold=1)[0],
        neatnet.continuity.continuity(streets.copy())[0],
    )

    observed_strokes = observed[0]
    observed_c_ = observed[1]
    observed_e_ = observed[2]
    observed_s_ = observed[3]

    assert observed_strokes == known_strokes
    assert observed_c_ == known_c_
    assert observed_e_ == known_e_
    assert observed_s_ == known_s_
