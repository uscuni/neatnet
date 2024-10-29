import copy
import itertools

import geopandas.testing
import momepy
import numpy
import pandas
import pytest
import shapely

import sgeop

crs = "EPSG:3857"

point_1 = shapely.Point(1, 1)
point_2 = shapely.Point(2, 2)
point_3 = shapely.Point(3, 3)
point_4 = shapely.Point(4, 4)
point_5 = shapely.Point(5, 5)
point_6 = shapely.Point(6, 6)
point_7 = shapely.Point(7, 7)
point_8 = shapely.Point(8, 8)
point_9 = shapely.Point(9, 9)
point_24 = shapely.Point(2, 4)
point_42 = shapely.Point(4, 2)

split_list_2 = [point_2]
split_array_2 = numpy.array(split_list_2)
split_series_2 = geopandas.GeoSeries(split_array_2)

split_list_3 = [point_3]
split_array_3 = numpy.array(split_list_3)
split_series_3 = geopandas.GeoSeries(split_array_3)

split_list_2_3 = split_list_2 + split_list_3
split_array_2_3 = numpy.array(split_list_2_3)
split_series_2_3 = geopandas.GeoSeries(split_array_2_3)

split_list_2_8 = split_list_2 + [point_8]
split_array_2_8 = numpy.array(split_list_2_8)
split_series_2_8 = geopandas.GeoSeries(split_array_2_8)

split_list_2_3_7_8 = split_list_2_3 + [point_7, point_8]
split_array_2_3_7_8 = numpy.array(split_list_2_3_7_8)
split_series_2_3_7_8 = geopandas.GeoSeries(split_array_2_3_7_8)

line_1_4 = shapely.LineString((point_1, point_4))
line_1_2 = shapely.LineString((point_1, point_2))
line_1_3 = shapely.LineString((point_1, point_3))
line_2_3 = shapely.LineString((point_2, point_3))
line_2_4 = shapely.LineString((point_2, point_4))
line_3_4 = shapely.LineString((point_3, point_4))
line_6_9 = shapely.LineString((point_6, point_9))
line_6_7 = shapely.LineString((point_6, point_7))
line_6_8 = shapely.LineString((point_6, point_8))
line_7_8 = shapely.LineString((point_7, point_8))
line_8_9 = shapely.LineString((point_8, point_9))
line_24_42 = shapely.LineString((point_24, point_42))
line_24_3 = shapely.LineString((point_24, point_3))
line_3_42 = shapely.LineString((point_3, point_42))

cases = range(1, 9)
types = ["list", "array", "series"]

# case 1: 1 road input -- not split
cleaned_roads_1 = geopandas.GeoDataFrame(geometry=[line_1_2], crs=crs)
known_1 = cleaned_roads_1.copy()

# case 2: 1 road input -- split once
cleaned_roads_2 = geopandas.GeoDataFrame(geometry=[line_1_4], crs=crs)
known_2 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed"]},
    geometry=[line_1_2, line_2_4],
    crs=crs,
)

# case 3: 1 road input -- split twice
cleaned_roads_3 = geopandas.GeoDataFrame(geometry=[line_1_4], crs=crs)
known_3 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed"]},
    geometry=[line_1_2, line_2_3, line_3_4],
    crs=crs,
)

# case 4: 2 roads input -- neither roads split
cleaned_roads_4 = geopandas.GeoDataFrame(geometry=[line_1_2, line_2_4], crs=crs)
known_4 = cleaned_roads_4.copy()

# case 5: 2 roads input -- 1 road split once
cleaned_roads_5 = geopandas.GeoDataFrame(geometry=[line_1_4, line_6_9], crs=crs)
known_5 = geopandas.GeoDataFrame(
    {"_status": [numpy.nan, "changed", "changed"]},
    geometry=[line_6_9, line_1_2, line_2_4],
    crs=crs,
)

# case 6: 2 roads input -- 2 roads split once (unique splits)
cleaned_roads_6 = cleaned_roads_5.copy()
known_6 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed", "changed"]},
    geometry=[line_1_2, line_2_4, line_6_8, line_8_9],
    crs=crs,
)

# case 7: 2 roads input -- 2 roads split twice (unique splits)
cleaned_roads_7 = cleaned_roads_5.copy()
known_7 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed", "changed", "changed", "changed"]},
    geometry=[line_1_2, line_2_3, line_3_4, line_6_7, line_7_8, line_8_9],
    crs=crs,
)

# case 8: 2 roads input (perpendicular)-- 2 roads split once (intersection)
cleaned_roads_8 = geopandas.GeoDataFrame(geometry=[line_1_4, line_24_42], crs=crs)
known_8 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed", "changed"]},
    geometry=[line_1_3, line_3_4, line_24_3, line_3_42],
    crs=crs,
)


@pytest.mark.parametrize(
    "split_points,cleaned_roads,known",
    (
        [split_list_2, cleaned_roads_1, known_1],  # case 1
        [split_array_2, cleaned_roads_1, known_1],
        [split_series_2, cleaned_roads_1, known_1],
        [split_list_2, cleaned_roads_2, known_2],  # case 2
        [split_array_2, cleaned_roads_2, known_2],
        [split_series_2, cleaned_roads_2, known_2],
        [split_list_2_3, cleaned_roads_3, known_3],  # case 3
        [split_array_2_3, cleaned_roads_3, known_3],
        [split_series_2_3, cleaned_roads_3, known_3],
        [split_list_2, cleaned_roads_4, known_4],  # case 4
        [split_array_2, cleaned_roads_4, known_4],
        [split_series_2, cleaned_roads_4, known_4],
        [split_list_2, cleaned_roads_5, known_5],  # case 5
        [split_array_2, cleaned_roads_5, known_5],
        [split_series_2, cleaned_roads_5, known_5],
        [split_list_2_8, cleaned_roads_6, known_6],  # case 6
        [split_array_2_8, cleaned_roads_6, known_6],
        [split_series_2_8, cleaned_roads_6, known_6],
        [split_list_2_3_7_8, cleaned_roads_7, known_7],  # case 7
        [split_array_2_3_7_8, cleaned_roads_7, known_7],
        [split_series_2_3_7_8, cleaned_roads_7, known_7],
        [split_list_3, cleaned_roads_8, known_8],  # case 8
        [split_array_3, cleaned_roads_8, known_8],
        [split_series_3, cleaned_roads_8, known_8],
    ),
    ids=[f"case{c}-{t}" for c, t in list(itertools.product(cases, types))],
)
def test_split(split_points, cleaned_roads, known):
    observed = sgeop.nodes.split(split_points, cleaned_roads, crs)
    assert isinstance(observed, geopandas.GeoDataFrame)
    assert observed.crs == known.crs == cleaned_roads.crs == crs
    pytest.geom_test(observed.geometry, known.geometry)
    if "_status" in observed.columns:
        pandas.testing.assert_series_equal(observed["_status"], known["_status"])


point_20001 = shapely.Point(2.0001, 2.0001)
point_21 = shapely.Point(2.1, 2.1)

line_1_20001 = shapely.LineString((point_1, point_20001))
line_20001_4 = shapely.LineString((point_20001, point_4))
line_1_21 = shapely.LineString((point_1, point_21))
line_21_4 = shapely.LineString((point_21, point_4))
line_1_6 = shapely.LineString((point_1, point_6))


@pytest.mark.parametrize(
    "edge,split_point,tol,known",
    (
        [line_1_4, point_2, 0.0000001, numpy.array([line_1_2, line_2_4])],
        [line_1_4, point_20001, 0.0000001, numpy.array([line_1_20001, line_20001_4])],
        [line_1_4, point_21, 0.0001, numpy.array([line_1_21, line_21_4])],
        [line_1_4, point_6, 0.1, numpy.array([line_1_4])],
        [line_1_4, point_6, 3, numpy.array([line_1_6])],
    ),
    ids=["exact", "precise", "relaxed", "ignore", "extend"],
)
def test_snap_n_split(edge, split_point, tol, known):
    observed = sgeop.nodes._snap_n_split(edge, split_point, tol)
    numpy.testing.assert_array_equal(observed, known)


line_3_4 = shapely.LineString((point_3, point_4))
line_4_5 = shapely.LineString((point_4, point_5))
line_234 = shapely.LineString((point_2, point_3, point_4))

edgeline_types_get_components = [
    [line_1_2, line_2_4],
    numpy.array([line_1_2, line_3_4]),
    geopandas.GeoSeries([line_1_2, line_234]),
    [line_1_2, line_2_4] + [line_4_5],
]

ignore_types_get_components = [
    None,
    point_2,
    [point_2],
    numpy.array([point_3]),
    geopandas.GeoSeries([point_3]),
]

cases_types_get_components = [
    list(c)
    for c in itertools.product(
        edgeline_types_get_components, ignore_types_get_components
    )
]

known_get_components = [
    [0, 0],
    [2.0, 3.0],
    [2.0, 3.0],
    [0, 0],
    [0, 0],
    [2.0, 3.0],
    [2.0, 3.0],
    [2.0, 3.0],
    [2.0, 3.0],
    [2.0, 3.0],
    [0, 0],
    [2.0, 3.0],
    [2.0, 3.0],
    [0, 0],
    [0, 0],
    [0, 0, 0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0, 0, 0],
    [0, 0, 0],
]

cases_get_components = [
    (*arg12, arg3)
    for arg12, arg3 in list(
        zip(cases_types_get_components, known_get_components, strict=True)
    )
]


t1_get_components = ["list", "ndarray", "GeoSeries", "list"]
t2_get_components = ["NoneType", "Point", "list", "ndarray", "GeoSeries"]
case_ids_get_components = [
    "-".join(c) for c in itertools.product(t1_get_components, t2_get_components)
]


@pytest.mark.parametrize(
    "edgelines,ignore,known",
    cases_get_components,
    ids=case_ids_get_components,
)
def test_get_components(edgelines, ignore, known):
    observed = sgeop.nodes.get_components(edgelines, ignore=ignore)
    numpy.testing.assert_array_equal(observed, known)


line_124 = shapely.LineString((point_1, point_2, point_4))
line_1234 = shapely.LineString((point_1, point_2, point_3, point_4))
line_245 = shapely.LineString((point_2, point_4, point_5))
line_1245 = shapely.LineString((point_1, point_2, point_4, point_5))

known_weld_edges = [
    [line_124],
    [line_1_2, line_2_4],
    [line_1_2, line_2_4],
    [line_124],
    [line_124],
    [line_1_2, line_3_4],
    [line_1_2, line_3_4],
    [line_1_2, line_3_4],
    [line_1_2, line_3_4],
    [line_1_2, line_3_4],
    [line_1234],
    [line_1_2, line_234],
    [line_1_2, line_234],
    [line_1234],
    [line_1234],
    [line_1245],
    [line_245, line_1_2],
    [line_245, line_1_2],
    [line_1245],
    [line_1245],
]

cases_types_weld_edges = copy.deepcopy(cases_types_get_components)


cases_weld_edges = [
    (*arg12, arg3)
    for arg12, arg3 in list(zip(cases_types_weld_edges, known_weld_edges, strict=True))
]

case_ids_weld_edges = copy.deepcopy(case_ids_get_components)


@pytest.mark.parametrize(
    "edgelines,ignore,known",
    cases_weld_edges,
    ids=case_ids_weld_edges,
)
def test_weld_edges(edgelines, ignore, known):
    observed = sgeop.nodes.weld_edges(edgelines, ignore=ignore)
    numpy.testing.assert_array_equal(observed, known)


class TestRemoveFalseNodes:
    def setup_method(self):
        p10 = shapely.Point(1, 0)
        p20 = shapely.Point(2, 0)
        p30 = shapely.Point(3, 0)
        p40 = shapely.Point(4, 0)
        p50 = shapely.Point(5, 0)
        p21 = shapely.Point(2, 1)
        p32 = shapely.Point(3, 2)
        p41 = shapely.Point(4, 1)

        self.line1020 = shapely.LineString((p10, p20))
        self.line2030 = shapely.LineString((p20, p30))
        self.line3040 = shapely.LineString((p30, p40))
        self.line4050 = shapely.LineString((p40, p50))
        self.line3021 = shapely.LineString((p30, p21))
        self.line2132 = shapely.LineString((p21, p32))
        self.line4132 = shapely.LineString((p41, p32))
        self.line3041 = shapely.LineString((p30, p41))
        self.attrs = ["cat"] * 3 + ["dog"] * 3 + ["eel"] * 2

        self.series = geopandas.GeoSeries(
            [
                self.line1020,
                self.line2030,
                self.line3040,
                self.line4050,
                self.line3021,
                self.line2132,
                self.line4132,
                self.line3041,
            ]
        )

        self.line102030 = shapely.LineString((p10, p20, p30))
        self.line304050 = shapely.LineString((p30, p40, p50))
        self.line3041323130 = shapely.LineString((p30, p41, p32, p21, p30))

        self.known_geoms = [
            self.line102030,
            self.line3041323130,
            self.line304050,
        ]

    def test_single_series(self):
        one_in_series = self.series[:0].copy()
        known = one_in_series
        observed = sgeop.nodes.remove_false_nodes(one_in_series)
        geopandas.testing.assert_geoseries_equal(observed, known)

    def test_series(self):
        known = geopandas.GeoDataFrame(geometry=self.known_geoms)
        observed = sgeop.nodes.remove_false_nodes(self.series)
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_frame(self):
        known = geopandas.GeoDataFrame(geometry=self.known_geoms)
        observed = sgeop.nodes.remove_false_nodes(
            geopandas.GeoDataFrame(geometry=self.series)
        )
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_frame_attrs_first(self):
        known = geopandas.GeoDataFrame(
            {"animal": ["cat", "dog", "cat"]},
            geometry=self.known_geoms,
            columns=["geometry", "animal"],
        )
        observed = sgeop.nodes.remove_false_nodes(
            geopandas.GeoDataFrame({"animal": self.attrs}, geometry=self.series)
        )
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_frame_attrs_last(self):
        known = geopandas.GeoDataFrame(
            {"animal": ["cat", "eel", "dog"]},
            geometry=self.known_geoms,
            columns=["geometry", "animal"],
        )
        observed = sgeop.nodes.remove_false_nodes(
            geopandas.GeoDataFrame({"animal": self.attrs}, geometry=self.series),
            aggfunc="last",
        )
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_momepy_suite(self):
        false_network = geopandas.read_file(
            momepy.datasets.get_path("tests"), layer="network"
        )
        false_network["vals"] = range(len(false_network))
        fixed = sgeop.remove_false_nodes(false_network).reset_index(drop=True)
        assert len(fixed) == 56
        assert isinstance(fixed, geopandas.GeoDataFrame)
        assert false_network.crs.equals(fixed.crs)
        assert sorted(false_network.columns) == sorted(fixed.columns)

        # check loop order
        expected = numpy.array(
            [
                [-727238.49292668, -1052817.28071986],
                [-727253.1752498, -1052827.47329062],
                [-727223.93217677, -1052829.47624082],
                [-727238.49292668, -1052817.28071986],
            ]
        )
        numpy.testing.assert_almost_equal(
            numpy.array(fixed.loc[55].geometry.coords), expected
        )

        fixed_series = sgeop.nodes.remove_false_nodes(
            false_network.geometry
        ).reset_index(drop=True)
        assert len(fixed_series) == 56
        assert isinstance(fixed_series, geopandas.GeoDataFrame)
        assert false_network.crs.equals(fixed_series.crs)

        multiindex = false_network.explode(index_parts=True)
        fixed_multiindex = sgeop.nodes.remove_false_nodes(multiindex)
        assert len(fixed_multiindex) == 56
        assert isinstance(fixed, geopandas.GeoDataFrame)
        assert sorted(false_network.columns) == sorted(fixed.columns)

        # no node of a degree 2
        df_streets = geopandas.read_file(
            momepy.datasets.get_path("bubenec"), layer="streets"
        )
        known = df_streets.drop([4, 7, 17, 22]).reset_index(drop=True)
        observed = sgeop.nodes.remove_false_nodes(known).reset_index(drop=True)
        geopandas.testing.assert_geodataframe_equal(observed, known)
