import copy
import itertools

import geopandas.testing
import momepy
import numpy
import pandas
import pytest
import shapely
from pandas.testing import assert_frame_equal, assert_series_equal

import neatnet

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

# case 1: 1 street input -- not split
cleaned_streets_1 = geopandas.GeoDataFrame(geometry=[line_1_2], crs=crs)
known_1 = cleaned_streets_1.copy()

# case 2: 1 street input -- split once
cleaned_streets_2 = geopandas.GeoDataFrame(geometry=[line_1_4], crs=crs)
known_2 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed"]},
    geometry=[line_1_2, line_2_4],
    crs=crs,
)

# case 3: 1 street input -- split twice
cleaned_streets_3 = geopandas.GeoDataFrame(geometry=[line_1_4], crs=crs)
known_3 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed"]},
    geometry=[line_1_2, line_2_3, line_3_4],
    crs=crs,
)

# case 4: 2 streets input -- neither streets split
cleaned_streets_4 = geopandas.GeoDataFrame(geometry=[line_1_2, line_2_4], crs=crs)
known_4 = cleaned_streets_4.copy()

# case 5: 2 street input -- 1 street split once
cleaned_streets_5 = geopandas.GeoDataFrame(geometry=[line_1_4, line_6_9], crs=crs)
known_5 = geopandas.GeoDataFrame(
    {"_status": [numpy.nan, "changed", "changed"]},
    geometry=[line_6_9, line_1_2, line_2_4],
    crs=crs,
)

# case 6: 2 streets input -- 2 streets split once (unique splits)
cleaned_streets_6 = cleaned_streets_5.copy()
known_6 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed", "changed"]},
    geometry=[line_1_2, line_2_4, line_6_8, line_8_9],
    crs=crs,
)

# case 7: 2 street input -- 2 streets split twice (unique splits)
cleaned_streets_7 = cleaned_streets_5.copy()
known_7 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed", "changed", "changed", "changed"]},
    geometry=[line_1_2, line_2_3, line_3_4, line_6_7, line_7_8, line_8_9],
    crs=crs,
)

# case 8: 2 streets input (perpendicular)-- 2 streets split once (intersection)
cleaned_streets_8 = geopandas.GeoDataFrame(geometry=[line_1_4, line_24_42], crs=crs)
known_8 = geopandas.GeoDataFrame(
    {"_status": ["changed", "changed", "changed", "changed"]},
    geometry=[line_1_3, line_3_4, line_24_3, line_3_42],
    crs=crs,
)


@pytest.mark.parametrize(
    "split_points,cleaned_streets,known",
    (
        [split_list_2, cleaned_streets_1, known_1],  # case 1
        [split_array_2, cleaned_streets_1, known_1],
        [split_series_2, cleaned_streets_1, known_1],
        [split_list_2, cleaned_streets_2, known_2],  # case 2
        [split_array_2, cleaned_streets_2, known_2],
        [split_series_2, cleaned_streets_2, known_2],
        [split_list_2_3, cleaned_streets_3, known_3],  # case 3
        [split_array_2_3, cleaned_streets_3, known_3],
        [split_series_2_3, cleaned_streets_3, known_3],
        [split_list_2, cleaned_streets_4, known_4],  # case 4
        [split_array_2, cleaned_streets_4, known_4],
        [split_series_2, cleaned_streets_4, known_4],
        [split_list_2, cleaned_streets_5, known_5],  # case 5
        [split_array_2, cleaned_streets_5, known_5],
        [split_series_2, cleaned_streets_5, known_5],
        [split_list_2_8, cleaned_streets_6, known_6],  # case 6
        [split_array_2_8, cleaned_streets_6, known_6],
        [split_series_2_8, cleaned_streets_6, known_6],
        [split_list_2_3_7_8, cleaned_streets_7, known_7],  # case 7
        [split_array_2_3_7_8, cleaned_streets_7, known_7],
        [split_series_2_3_7_8, cleaned_streets_7, known_7],
        [split_list_3, cleaned_streets_8, known_8],  # case 8
        [split_array_3, cleaned_streets_8, known_8],
        [split_series_3, cleaned_streets_8, known_8],
    ),
    ids=[f"case{c}-{t}" for c, t in list(itertools.product(cases, types))],
)
def test_split(split_points, cleaned_streets, known):
    observed = neatnet.nodes.split(split_points, cleaned_streets, crs)
    assert isinstance(observed, geopandas.GeoDataFrame)
    assert observed.crs == known.crs == cleaned_streets.crs == crs
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
    observed = neatnet.nodes._snap_n_split(edge, split_point, tol)
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
    observed = neatnet.nodes.get_components(edgelines, ignore=ignore)
    numpy.testing.assert_array_equal(observed, known)


class TestBowtie:
    # see gh:214

    def setup_method(self):
        p00, p01 = shapely.Point(1, 0), shapely.Point(1, 1)
        p02, p03 = shapely.Point(1, 3), shapely.Point(1, 5)
        p04, p05 = shapely.Point(3, 3), shapely.Point(3, 1)
        p06, p07 = shapely.Point(0, 1), shapely.Point(5, 1)

        l00 = shapely.LineString((p00, p01))
        l01 = shapely.LineString((p01, p02))
        l02 = shapely.LineString((p02, p03))
        l03 = shapely.LineString((p02, p04))
        l04 = shapely.LineString((p04, p05))
        l05 = shapely.LineString((p06, p01))
        l06 = shapely.LineString((p01, p05))
        l07 = shapely.LineString((p05, p07))

        # bowtie ----------------------------------------------
        # left edge loop points
        b01, b02, b03 = shapely.Point(6, 1), shapely.Point(6, 3), shapely.Point(7, 2)

        # right edge loop points
        b04, b05, b06 = shapely.Point(12, 1), shapely.Point(12, 3), shapely.Point(11, 2)

        # two middle edges points
        b07, b08 = shapely.Point(8, 3), shapely.Point(10, 3)
        b09, b10 = shapely.Point(10, 1), shapely.Point(8, 1)

        # left and right loops
        loop1 = shapely.LineString((b03, b01, b02, b03))
        loop2 = shapely.LineString((b06, b04, b05, b06))

        # lower & upper middle edges
        mid1 = shapely.LineString((b03, b07, b08, b06))
        mid2 = shapely.LineString((b06, b09, b10, b03))

        edges = geopandas.GeoDataFrame(
            geometry=[l00, l01, l02, l03, l04, l05, l06, l07, loop1, loop2, mid1, mid2]
        )

        self.edges = edges.explode(ignore_index=True).geometry
        self.bowtie_nodes = geopandas.GeoSeries([b03, b06])

        self.known_comp_labels = numpy.array(
            [1.0, 2.0, 3.0, 0.0, 0.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )

    def test_get_components_bowtie(self):
        known = self.known_comp_labels
        observed = neatnet.nodes.get_components(self.edges)
        numpy.testing.assert_array_equal(observed, known)

    def test_get_components_bowtie_ignore(self):
        known = self.known_comp_labels
        observed = neatnet.nodes.get_components(self.edges, ignore=self.bowtie_nodes)
        numpy.testing.assert_array_equal(observed, known)

    def test_isolate_bowtie_nodes_list(self):
        known = self.bowtie_nodes
        observed = neatnet.nodes.isolate_bowtie_nodes(self.edges.geometry.tolist())
        geopandas.testing.assert_geoseries_equal(observed, known)

    def test_isolate_bowtie_nodes_array(self):
        known = self.bowtie_nodes
        observed = neatnet.nodes.isolate_bowtie_nodes(numpy.array(self.edges.geometry))
        geopandas.testing.assert_geoseries_equal(observed, known)

    def test_isolate_bowtie_nodes_geoseries(self):
        known = self.bowtie_nodes
        observed = neatnet.nodes.isolate_bowtie_nodes(self.edges.geometry)
        geopandas.testing.assert_geoseries_equal(observed, known)

    def test_isolate_bowtie_nodes_none(self):
        known = geopandas.GeoSeries([])
        observed = neatnet.nodes.isolate_bowtie_nodes(self.edges.geometry[:8])
        geopandas.testing.assert_geoseries_equal(observed, known)


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
    observed = neatnet.nodes.weld_edges(edgelines, ignore=ignore)
    numpy.testing.assert_array_equal(observed, known)


class TestInduceNodes:
    def setup_method(self):
        self.p10 = shapely.Point(1, 0)
        self.p20 = shapely.Point(2, 0)
        self.p201 = shapely.Point(2, 0.1)
        self.p30 = shapely.Point(3, 0)
        self.p40 = shapely.Point(4, 0)
        self.p21 = shapely.Point(2, 1)
        self.p41 = shapely.Point(4, 1)
        self.p251 = shapely.Point(2.5, 1)
        self.p215 = shapely.Point(2, 1.5)
        self.p315 = shapely.Point(3, 1.5)

        self.line1020 = shapely.LineString((self.p10, self.p20))
        self.line1030 = shapely.LineString((self.p10, self.p30))
        self.line20121 = shapely.LineString((self.p201, self.p21))
        self.line2021 = shapely.LineString((self.p20, self.p21))
        self.line2030 = shapely.LineString((self.p20, self.p30))
        self.line2040 = shapely.LineString((self.p20, self.p40))
        self.line215351251 = shapely.LineString((self.p215, self.p315, self.p251))
        self.line251215 = shapely.LineString((self.p251, self.p215))
        self.line25141 = shapely.LineString((self.p251, self.p41))
        self.line2514130 = shapely.LineString((self.p251, self.p41, self.p30))
        self.line3040 = shapely.LineString((self.p30, self.p40))
        self.line3021251 = shapely.LineString((self.p30, self.p21, self.p251))
        self.line3041 = shapely.LineString((self.p30, self.p41))
        self.line4130 = shapely.LineString((self.p41, self.p30))
        self.line413021 = shapely.LineString((self.p41, self.p30, self.p21))
        self.line41302141 = shapely.LineString((self.p41, self.p30, self.p21, self.p41))
        self.line30214130 = shapely.LineString((self.p30, self.p21, self.p41, self.p30))
        self.line215315251215 = shapely.LineString(
            (self.p215, self.p315, self.p251, self.p215)
        )
        self.line251215315251 = shapely.LineString(
            (self.p251, self.p215, self.p315, self.p251)
        )

    def test_induced_simple(self):
        known = geopandas.GeoDataFrame(
            {
                "geometry": [self.line2021, self.line1020, self.line2030],
                "_status": [numpy.nan, "changed", "changed"],
            }
        )
        edges = geopandas.GeoDataFrame(geometry=[self.line1030, self.line2021])
        observed = neatnet.nodes.induce_nodes(edges)
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_not_induced_simple(self):
        known = geopandas.GeoDataFrame(geometry=[self.line1030, self.line20121])
        edges = geopandas.GeoDataFrame(geometry=[self.line1030, self.line20121])
        observed = neatnet.nodes.induce_nodes(edges)
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_induced_complex(self):
        known = geopandas.GeoDataFrame(
            {
                "geometry": [
                    self.line2030,
                    self.line3040,
                    self.line4130,
                    self.line3021251,
                    self.line25141,
                    self.line215351251,
                    self.line251215,
                ],
                "_status": ["changed"] * 7,
            }
        )
        edges = geopandas.GeoDataFrame(
            geometry=[self.line2040, self.line41302141, self.line215315251215]
        )
        observed = neatnet.nodes.induce_nodes(edges)
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_not_induced_complex(self):
        known = geopandas.GeoDataFrame(
            {
                "geometry": [
                    self.line2030,
                    self.line3040,
                    self.line3021251,
                    self.line2514130,
                    self.line251215315251,
                ],
            }
        )
        edges = geopandas.GeoDataFrame(
            geometry=[
                self.line2030,
                self.line3040,
                self.line3021251,
                self.line2514130,
                self.line251215315251,
            ]
        )
        observed = neatnet.nodes.induce_nodes(edges)
        geopandas.testing.assert_geodataframe_equal(observed, known)


class TestIdentifyDegreeMismatch:
    def setup_method(self):
        self.p20 = shapely.Point(2, 0)
        self.p30 = shapely.Point(3, 0)
        self.p40 = shapely.Point(4, 0)
        self.p21 = shapely.Point(2, 1)
        self.p41 = shapely.Point(4, 1)

        self.line2040 = shapely.LineString((self.p20, self.p40))
        self.line413021 = shapely.LineString((self.p41, self.p30, self.p21))
        self.line41302141 = shapely.LineString((self.p41, self.p30, self.p21, self.p41))

        self.sindex_kws = {"predicate": "dwithin", "distance": 1e-4}

    def test_no_mismatch(self):
        known = geopandas.GeoSeries([])
        edges = geopandas.GeoDataFrame(geometry=[self.line2040, self.line413021])
        observed = neatnet.nodes._identify_degree_mismatch(edges, self.sindex_kws)
        geopandas.testing.assert_geoseries_equal(observed, known)

    def test_mismatch(self):
        known = geopandas.GeoSeries([self.p41], index=[2])
        edges = geopandas.GeoDataFrame(geometry=[self.line2040, self.line41302141])
        observed = neatnet.nodes._identify_degree_mismatch(edges, self.sindex_kws)
        geopandas.testing.assert_geoseries_equal(observed, known)


class TestMakesLoopContact:
    def setup_method(self):
        self.p20 = shapely.Point(2, 0)
        self.p30 = shapely.Point(3, 0)
        self.p40 = shapely.Point(4, 0)
        self.p21 = shapely.Point(2, 1)
        self.p41 = shapely.Point(4, 1)
        self.p251 = shapely.Point(2.5, 1)
        self.p215 = shapely.Point(2, 1.5)
        self.p315 = shapely.Point(3, 1.5)

        self.line2040 = shapely.LineString((self.p20, self.p40))
        self.line2030 = shapely.LineString((self.p20, self.p30))
        self.line3040 = shapely.LineString((self.p30, self.p40))
        self.line3041 = shapely.LineString((self.p30, self.p41))
        self.line413021 = shapely.LineString((self.p41, self.p30, self.p21))
        self.line41302141 = shapely.LineString((self.p41, self.p30, self.p21, self.p41))
        self.line30214130 = shapely.LineString((self.p30, self.p21, self.p41, self.p30))
        self.line215315251215 = shapely.LineString(
            (self.p215, self.p315, self.p251, self.p215)
        )
        self.line251215315251 = shapely.LineString(
            (self.p251, self.p215, self.p315, self.p251)
        )

        self.sindex_kws = {"predicate": "dwithin", "distance": 1e-4}

    def test_off_1_bad_order(self):
        known_off_loops = geopandas.GeoSeries([self.p30], index=[1])
        known_on_loops = geopandas.GeoSeries([])
        edges = geopandas.GeoDataFrame(geometry=[self.line2040, self.line41302141])
        observed_off_loops, observed_on_loops = neatnet.nodes._makes_loop_contact(
            edges, self.sindex_kws
        )
        geopandas.testing.assert_geoseries_equal(observed_off_loops, known_off_loops)
        geopandas.testing.assert_geoseries_equal(observed_on_loops, known_on_loops)

    def test_off_2_bad_order(self):
        known_off_loops = geopandas.GeoSeries([self.p30, self.p30], index=[1, 1])
        known_on_loops = geopandas.GeoSeries([])
        edges = geopandas.GeoDataFrame(
            geometry=[self.line2030, self.line3040, self.line41302141]
        )
        observed_off_loops, observed_on_loops = neatnet.nodes._makes_loop_contact(
            edges, self.sindex_kws
        )
        geopandas.testing.assert_geoseries_equal(observed_off_loops, known_off_loops)
        geopandas.testing.assert_geoseries_equal(observed_on_loops, known_on_loops)

    def test_off_1_good_order(self):
        known_off_loops = geopandas.GeoSeries([self.p30, self.p30], index=[0, 3])
        known_on_loops = geopandas.GeoSeries([])
        edges = geopandas.GeoDataFrame(geometry=[self.line2040, self.line30214130])
        observed_off_loops, observed_on_loops = neatnet.nodes._makes_loop_contact(
            edges, self.sindex_kws
        )
        geopandas.testing.assert_geoseries_equal(observed_off_loops, known_off_loops)
        geopandas.testing.assert_geoseries_equal(observed_on_loops, known_on_loops)

    def test_off_2_good_order(self):
        known_off_loops = geopandas.GeoSeries([self.p30] * 4, index=[0, 0, 3, 3])
        known_on_loops = geopandas.GeoSeries([])
        edges = geopandas.GeoDataFrame(
            geometry=[self.line2030, self.line3040, self.line30214130]
        )
        observed_off_loops, observed_on_loops = neatnet.nodes._makes_loop_contact(
            edges, self.sindex_kws
        )
        geopandas.testing.assert_geoseries_equal(observed_off_loops, known_off_loops)
        geopandas.testing.assert_geoseries_equal(observed_on_loops, known_on_loops)

    def test_on_1_bad_order(self):
        known_off_loops = geopandas.GeoSeries([])
        known_on_loops = geopandas.GeoSeries([self.p251], index=[6])
        edges = geopandas.GeoDataFrame(
            geometry=[self.line41302141, self.line215315251215]
        )
        observed_off_loops, observed_on_loops = neatnet.nodes._makes_loop_contact(
            edges, self.sindex_kws
        )
        geopandas.testing.assert_geoseries_equal(observed_off_loops, known_off_loops)
        geopandas.testing.assert_geoseries_equal(observed_on_loops, known_on_loops)

    def test_on_1_good_order(self):
        known_off_loops = geopandas.GeoSeries([])
        known_on_loops = geopandas.GeoSeries([self.p251, self.p251], index=[4, 7])
        edges = geopandas.GeoDataFrame(
            geometry=[self.line41302141, self.line251215315251]
        )
        observed_off_loops, observed_on_loops = neatnet.nodes._makes_loop_contact(
            edges, self.sindex_kws
        )
        geopandas.testing.assert_geoseries_equal(observed_off_loops, known_off_loops)
        geopandas.testing.assert_geoseries_equal(observed_on_loops, known_on_loops)

    def test_multi_geoms(self):
        known_off_loops = geopandas.GeoSeries([self.p30], index=[1])
        known_on_loops = geopandas.GeoSeries([self.p251, self.p251], index=[4, 7])
        edges = geopandas.GeoDataFrame(
            geometry=[self.line2040, self.line41302141, self.line251215315251]
        )
        observed_off_loops, observed_on_loops = neatnet.nodes._makes_loop_contact(
            edges, self.sindex_kws
        )
        geopandas.testing.assert_geoseries_equal(observed_off_loops, known_off_loops)
        geopandas.testing.assert_geoseries_equal(observed_on_loops, known_on_loops)

    def test_no_loops(self):
        known_off_loops = geopandas.GeoSeries([])
        known_on_loops = geopandas.GeoSeries([])
        edges = geopandas.GeoDataFrame(geometry=[self.line2040, self.line3041])
        observed_off_loops, observed_on_loops = neatnet.nodes._makes_loop_contact(
            edges, self.sindex_kws
        )
        geopandas.testing.assert_geoseries_equal(observed_off_loops, known_off_loops)
        geopandas.testing.assert_geoseries_equal(observed_on_loops, known_on_loops)


class TestLoopsAndNonloops:
    def setup_method(self):
        self.p20 = shapely.Point(2, 0)
        self.p30 = shapely.Point(3, 0)
        self.p40 = shapely.Point(4, 0)
        self.p21 = shapely.Point(2, 1)
        self.p41 = shapely.Point(4, 1)
        self.p251 = shapely.Point(2.5, 1)
        self.p215 = shapely.Point(2, 1.5)
        self.p315 = shapely.Point(3, 1.5)

        self.line2040 = shapely.LineString((self.p20, self.p40))
        self.line413021 = shapely.LineString((self.p41, self.p30, self.p21))
        self.line41302141 = shapely.LineString((self.p41, self.p30, self.p21, self.p41))
        self.line251215315251 = shapely.LineString(
            (self.p215, self.p315, self.p251, self.p215)
        )

    def test_only_loops(self):
        known_loops = geopandas.GeoDataFrame(
            geometry=[self.line41302141, self.line251215315251]
        )
        known_non_loops = geopandas.GeoDataFrame(geometry=[])
        edges = geopandas.GeoDataFrame(
            geometry=[self.line41302141, self.line251215315251]
        )
        observed_loops, observed_non_loops = neatnet.nodes._loops_and_non_loops(edges)

        geopandas.testing.assert_geodataframe_equal(observed_loops, known_loops)
        geopandas.testing.assert_geodataframe_equal(observed_non_loops, known_non_loops)

    def test_both(self):
        known_loops = geopandas.GeoDataFrame(geometry=[self.line41302141], index=[1])
        known_non_loops = geopandas.GeoDataFrame(geometry=[self.line2040])
        edges = geopandas.GeoDataFrame(geometry=[self.line2040, self.line41302141])
        observed_loops, observed_non_loops = neatnet.nodes._loops_and_non_loops(edges)

        geopandas.testing.assert_geodataframe_equal(observed_loops, known_loops)
        geopandas.testing.assert_geodataframe_equal(observed_non_loops, known_non_loops)

    def test_only_non_loops(self):
        known_loops = geopandas.GeoDataFrame(geometry=[])
        known_non_loops = geopandas.GeoDataFrame(
            geometry=[self.line2040, self.line413021]
        )
        edges = geopandas.GeoDataFrame(geometry=[self.line2040, self.line413021])
        observed_loops, observed_non_loops = neatnet.nodes._loops_and_non_loops(edges)

        geopandas.testing.assert_geodataframe_equal(observed_loops, known_loops)
        geopandas.testing.assert_geodataframe_equal(observed_non_loops, known_non_loops)


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
        observed = neatnet.nodes.remove_interstitial_nodes(one_in_series)
        geopandas.testing.assert_geoseries_equal(observed, known)

    def test_series(self):
        known = geopandas.GeoDataFrame(geometry=self.known_geoms)
        observed = neatnet.nodes.remove_interstitial_nodes(self.series)
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_frame(self):
        known = geopandas.GeoDataFrame(geometry=self.known_geoms)
        observed = neatnet.nodes.remove_interstitial_nodes(
            geopandas.GeoDataFrame(geometry=self.series)
        )
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_frame_attrs_first(self):
        known = geopandas.GeoDataFrame(
            {"animal": ["cat", "dog", "cat"]},
            geometry=self.known_geoms,
            columns=["geometry", "animal"],
        )
        observed = neatnet.nodes.remove_interstitial_nodes(
            geopandas.GeoDataFrame({"animal": self.attrs}, geometry=self.series)
        )
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_frame_attrs_last(self):
        known = geopandas.GeoDataFrame(
            {"animal": ["cat", "eel", "dog"]},
            geometry=self.known_geoms,
            columns=["geometry", "animal"],
        )
        observed = neatnet.nodes.remove_interstitial_nodes(
            geopandas.GeoDataFrame({"animal": self.attrs}, geometry=self.series),
            aggfunc="last",
        )
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_momepy_suite(self):
        false_network = geopandas.read_file(
            momepy.datasets.get_path("tests"), layer="network"
        )
        false_network["vals"] = range(len(false_network))
        fixed = neatnet.remove_interstitial_nodes(false_network).reset_index(drop=True)
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

        fixed_series = neatnet.nodes.remove_interstitial_nodes(
            false_network.geometry
        ).reset_index(drop=True)
        assert len(fixed_series) == 56
        assert isinstance(fixed_series, geopandas.GeoDataFrame)
        assert false_network.crs.equals(fixed_series.crs)

        multiindex = false_network.explode(index_parts=True)
        fixed_multiindex = neatnet.nodes.remove_interstitial_nodes(multiindex)
        assert len(fixed_multiindex) == 56
        assert isinstance(fixed, geopandas.GeoDataFrame)
        assert sorted(false_network.columns) == sorted(fixed.columns)

        # no node of a degree 2
        df_streets = geopandas.read_file(
            momepy.datasets.get_path("bubenec"), layer="streets"
        )
        known = df_streets.drop([4, 7, 17, 22]).reset_index(drop=True)
        observed = neatnet.nodes.remove_interstitial_nodes(known).reset_index(drop=True)
        geopandas.testing.assert_geodataframe_equal(observed, known)


class TestRotateLoopCoords:
    def setup_method(self):
        self.p20 = shapely.Point(2, 0)
        self.p30 = shapely.Point(3, 0)
        self.p40 = shapely.Point(4, 0)
        self.p21 = shapely.Point(2, 1)
        self.p41 = shapely.Point(4, 1)

        self.line2040 = shapely.LineString((self.p20, self.p40))
        self.line2030 = shapely.LineString((self.p20, self.p30))
        self.line3040 = shapely.LineString((self.p30, self.p40))
        self.line41302141 = shapely.LineString((self.p41, self.p30, self.p21, self.p41))
        self.line30214130 = shapely.LineString((self.p30, self.p21, self.p41, self.p30))

        self.known = shapely.LineString(
            numpy.array([[3.0, 0.0], [2.0, 1.0], [4.0, 1.0], [3.0, 0.0]])
        )

    def test_needs_rotate_intersects_1(self):
        edges = geopandas.GeoDataFrame(geometry=[self.line2040, self.line41302141])
        observed = neatnet.nodes._rotate_loop_coords(
            edges[edges.is_ring].geometry,
            edges[~edges.is_ring],
        )
        assert shapely.equals_exact(observed, self.known)

    def test_needs_rotate_intersects_2(self):
        edges = geopandas.GeoDataFrame(
            geometry=[self.line2030, self.line3040, self.line41302141]
        )
        observed = neatnet.nodes._rotate_loop_coords(
            edges[edges.is_ring].geometry,
            edges[~edges.is_ring],
        )
        assert shapely.equals_exact(observed, self.known)

    def test_no_rotate_intersects_1(self):
        edges = geopandas.GeoDataFrame(geometry=[self.line2040, self.line30214130])
        observed = neatnet.nodes._rotate_loop_coords(
            edges[edges.is_ring].geometry,
            edges[~edges.is_ring],
        )
        assert shapely.equals_exact(observed, self.known)

    def test_no_rotate_intersects_2(self):
        edges = geopandas.GeoDataFrame(
            geometry=[self.line2030, self.line3040, self.line30214130]
        )
        observed = neatnet.nodes._rotate_loop_coords(
            edges[edges.is_ring].geometry,
            edges[~edges.is_ring],
        )
        assert shapely.equals_exact(observed, self.known)

    def test_pathological_case(self):
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "id": "94354",
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": (
                            (429872.8265231161, 5403257.540722819),
                            (429876.5853363918, 5403260.2174394345),
                            (429890.4230388994, 5403270.350099772),
                            (429933.0335605647, 5403301.568361696),
                            (430020.5881361559, 5403365.644292961),
                            (430022.567668632, 5403366.475594611),
                        ),
                    },
                    "bbox": (
                        429872.8265231161,
                        5403257.540722819,
                        430022.567668632,
                        5403366.475594611,
                    ),
                },
                {
                    "id": "94357",
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": (
                            (429953.9200473605, 5403493.275151249),
                            (429956.8841244645, 5403489.413576888),
                            (430007.0080729545, 5403420.123804592),
                            (430010.5893164184, 5403416.254540715),
                            (430015.97709030996, 5403409.372039658),
                            (430017.4790092487, 5403407.852373851),
                            (430025.04614706535, 5403396.628982025),
                            (430026.44617518806, 5403394.599181716),
                            (430038.62751364004, 5403378.081702546),
                        ),
                    },
                    "bbox": (
                        429953.9200473605,
                        5403378.081702546,
                        430038.62751364004,
                        5403493.275151249,
                    ),
                },
                {
                    "id": "94372",
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": (
                            (430020.14518559794, 5403370.763945371),
                            (430022.567668632, 5403366.475594611),
                            (430025.95408123883, 5403362.886719342),
                            (430031.2922934975, 5403359.673674717),
                            (430037.3578635884, 5403358.219244937),
                            (430043.57943603944, 5403358.663987033),
                            (430049.375806696, 5403360.94846128),
                            (430054.2153914074, 5403364.868076147),
                            (430057.64547448995, 5403370.061605995),
                            (430059.33670441434, 5403376.043989073),
                            (430059.02531223604, 5403382.874109089),
                            (430056.466756139, 5403389.209789496),
                            (430051.95348833746, 5403394.346966572),
                            (430045.98532046797, 5403397.712397026),
                            (430039.2407325106, 5403398.919602621),
                            (430032.4749992514, 5403397.847965369),
                            (430026.44617518806, 5403394.599181716),
                            (430022.7400172959, 5403390.832180555),
                            (430020.1183115003, 5403386.240031483),
                            (430018.7687499527, 5403381.14279707),
                            (430018.7761774767, 5403375.861826809),
                        ),
                    },
                    "bbox": (
                        430018.7687499527,
                        5403358.219244937,
                        430059.33670441434,
                        5403398.919602621,
                    ),
                },
                {
                    "id": "337955",
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "LineString",
                        "coordinates": (
                            (430018.7761774767, 5403375.861826809),
                            (430020.14518559794, 5403370.763945371),
                        ),
                    },
                    "bbox": (
                        430018.7761774767,
                        5403370.763945371,
                        430020.14518559794,
                        5403375.861826809,
                    ),
                },
            ],
            "bbox": (
                429872.8265231161,
                5403257.540722819,
                430059.33670441434,
                5403493.275151249,
            ),
        }
        gdf = geopandas.GeoDataFrame.from_features(data)
        # see gh#224
        with pytest.warns(
            UserWarning,
            match=(
                "Unable to sort modes: '<' not supported between "
                "instances of 'Point' and 'Point'"
            ),
        ):
            r = neatnet.fix_topology(gdf)
        assert r.shape[0] == 3


def test_fix_topology():
    p20 = shapely.Point(2, 0)
    p30 = shapely.Point(3, 0)
    p40 = shapely.Point(4, 0)
    p21 = shapely.Point(2, 1)
    p41 = shapely.Point(4, 1)

    p251 = shapely.Point(2.5, 1)
    p215 = shapely.Point(2, 1.5)
    p315 = shapely.Point(3, 1.5)

    p27508 = shapely.Point(2.75, 0.8)
    p32508 = shapely.Point(3.25, 0.8)
    p31 = shapely.Point(3, 1)

    line2040 = shapely.LineString((p20, p40))
    line41313041 = shapely.LineString((p41, p21, p30, p41))
    line251215315251 = shapely.LineString((p215, p315, p251, p215))
    line275083250831 = shapely.LineString((p27508, p32508, p31, p27508))

    known = geopandas.GeoDataFrame(
        {
            "geometry": [
                shapely.LineString((p251, p215, p315, p251)),
                shapely.LineString((p31, p27508, p32508, p31)),
                shapely.LineString((p30, p41, p31)),
                shapely.LineString((p20, p30)),
                shapely.LineString((p30, p40)),
                shapely.LineString((p251, p21, p30)),
                shapely.LineString((p31, p251)),
            ],
            "_status": ["changed"] * 7,
        }
    )

    observed = neatnet.nodes.fix_topology(
        geopandas.GeoDataFrame(
            geometry=[
                line2040,
                line2040,
                line41313041,
                line251215315251,
                line275083250831,
            ]
        )
    )

    geopandas.testing.assert_geodataframe_equal(observed, known)


class TestConsolidateNodes:
    def setup_method(self):
        self.p1 = shapely.Point(10, 10)
        self.p2 = shapely.Point(15, 15)
        self.p3 = shapely.Point(15, 15.5)
        self.p4 = shapely.Point(17, 16)
        self.p5 = shapely.Point(17, 15)
        self.p6 = shapely.Point(10, 20)
        self.p7 = shapely.Point(20, 20)
        self.p8 = shapely.Point(20, 10)

        self.line1 = shapely.LineString((self.p1, self.p2))
        self.line2 = shapely.LineString((self.p2, self.p3))
        self.line3 = shapely.LineString((self.p3, self.p4))
        self.line4 = shapely.LineString((self.p4, self.p5))
        self.line5 = shapely.LineString((self.p5, self.p2))
        self.line6 = shapely.LineString((self.p6, self.p3))
        self.line7 = shapely.LineString((self.p7, self.p4))
        self.line8 = shapely.LineString((self.p8, self.p5))

        self.lines_array = numpy.array(
            [
                self.line1,
                self.line2,
                self.line3,
                self.line4,
                self.line5,
                self.line6,
                self.line7,
                self.line8,
            ]
        )

        self.lines_series = geopandas.GeoSeries(self.lines_array)
        self.lines_gframe = geopandas.GeoDataFrame(geometry=self.lines_series)

    def test_array_only_ends(self):
        known = geopandas.GeoDataFrame(
            {"geometry": numpy.array([self.line1]), "_status": ["original"]}
        )
        observed = neatnet.nodes.consolidate_nodes(geopandas.GeoSeries([self.line1]))
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_series_only_ends(self):
        known = geopandas.GeoDataFrame(
            {"geometry": [self.line1], "_status": ["original"]}
        )
        observed = neatnet.nodes.consolidate_nodes(geopandas.GeoSeries([self.line1]))
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_frame_only_ends(self):
        known = geopandas.GeoDataFrame(
            {"geometry": [self.line1], "_status": ["original"]}
        )
        observed = neatnet.nodes.consolidate_nodes(
            geopandas.GeoDataFrame(geometry=[self.line1]), preserve_ends=True
        )
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_array_no_change(self):
        known = geopandas.GeoDataFrame(
            {"geometry": self.lines_array, "_status": ["original"] * 8}
        )
        observed = neatnet.nodes.consolidate_nodes(self.lines_array, tolerance=0.1)
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_series_no_change(self):
        known = geopandas.GeoDataFrame(
            {"geometry": self.lines_series, "_status": ["original"] * 8}
        )
        observed = neatnet.nodes.consolidate_nodes(self.lines_series, tolerance=0.1)
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_frame_no_change(self):
        known = geopandas.GeoDataFrame(
            {"geometry": self.lines_series, "_status": ["original"] * 8}
        )
        observed = neatnet.nodes.consolidate_nodes(self.lines_gframe, tolerance=0.1)
        geopandas.testing.assert_geodataframe_equal(observed, known)

    def test_t05_pe(self):
        _p1 = shapely.Point(14.81439916202902, 15.667040754173883)
        _p2 = shapely.Point(15, 15.25)
        _p3 = shapely.Point(14.823223304703363, 14.823223304703363)
        _p4 = shapely.Point(15.242243505690334, 15.560560876422583)
        _p5 = shapely.Point(15.25, 15)

        known = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.LineString((self.p6, _p1, _p2)),
                    shapely.LineString((self.p1, _p3, _p2)),
                    shapely.LineString((_p2, _p4, self.p4)),
                    shapely.LineString((self.p5, _p5, _p2)),
                    shapely.LineString((self.p4, self.p5)),
                    shapely.LineString((self.p7, self.p4)),
                    shapely.LineString((self.p8, self.p5)),
                ],
                "_status": ["changed"] * 4 + ["original"] * 3,
            }
        )
        observed = neatnet.nodes.consolidate_nodes(
            self.lines_gframe, tolerance=0.5, preserve_ends=True
        )

        assert_series_equal(known._status, observed._status)
        pytest.geom_test(known, observed, tolerance=0.000001)

    def test_t1(self):
        _p1 = shapely.Point(14.628798324058037, 15.834081508347767)
        _p2 = shapely.Point(15, 15.25)
        _p3 = shapely.Point(14.646446609406727, 14.646446609406727)
        _p4 = shapely.Point(15.484487011380669, 15.621121752845166)
        _p5 = shapely.Point(16.5, 15.875)
        _p6 = shapely.Point(17, 15.5)
        _p7 = shapely.Point(16.5, 15)
        _p8 = shapely.Point(15.5, 15)
        _p9 = shapely.Point(17.256938038358054, 14.571769936069908)
        _p10 = shapely.Point(17.299642949358844, 16.39952393247846)

        known = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.LineString((self.p6, _p1, _p2)),
                    shapely.LineString((self.p1, _p3, _p2)),
                    shapely.LineString((_p2, _p4, _p5, _p6)),
                    shapely.LineString((_p6, _p7, _p8, _p2)),
                    shapely.LineString((self.p8, _p9, _p6)),
                    shapely.LineString((self.p7, _p10, _p6)),
                ],
                "_status": ["changed"] * 6,
            }
        )
        observed = neatnet.nodes.consolidate_nodes(self.lines_gframe, tolerance=1)

        assert_series_equal(known._status, observed._status)
        pytest.geom_test(known, observed, tolerance=0.000001)

    def test_t2(self):
        known = geopandas.GeoDataFrame(
            {
                "geometry": geopandas.GeoSeries.from_wkt(
                    [
                        "LINESTRING (10 20, 14.257597 16.168163, 15 15.25)",
                        "LINESTRING (10 10, 14.292893 14.292893, 15 15.25)",
                        "LINESTRING (15 15.25, 15.968974 15.742244, 16 15.75, 17 15.5)",
                        "LINESTRING (15 15.25, 16 15, 17 15.5)",
                        "LINESTRING (20 10, 17.513876 14.14354, 17 15.5)",
                        "LINESTRING (20 20, 17.599286 16.799048, 17 15.5)",
                    ]
                ),
                "_status": ["changed"] * 6,
            }
        )
        observed = neatnet.nodes.consolidate_nodes(self.lines_gframe, tolerance=2)

        assert_series_equal(known._status, observed._status)
        pytest.geom_test(known, observed, tolerance=0.000001)

    def test_t5(self):
        _p1 = shapely.Point(16, 15.375)
        _p2 = shapely.Point(13.143991620290183, 17.170407541738836)
        _p3 = shapely.Point(13.232233047033631, 13.232233047033631)
        _p4 = shapely.Point(18.28469019179028, 12.858849680349534)
        _p5 = shapely.Point(18.498214746794215, 17.99761966239229)

        known = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.LineString((self.p6, _p2, _p1)),
                    shapely.LineString((self.p1, _p3, _p1)),
                    shapely.LineString((self.p8, _p4, _p1)),
                    shapely.LineString((self.p7, _p5, _p1)),
                ],
                "_status": ["changed"] * 4,
            }
        )
        observed = neatnet.nodes.consolidate_nodes(self.lines_gframe, tolerance=5)

        assert_series_equal(known._status, observed._status)
        pytest.geom_test(known, observed, tolerance=0.000001)

    def test_t5_pe(self):
        _p1 = shapely.Point(16, 15.375)
        _p2 = shapely.Point(13.143991620290183, 17.170407541738836)
        _p3 = shapely.Point(13.232233047033631, 13.232233047033631)
        _p4 = shapely.Point(18.28469019179028, 12.858849680349534)
        _p5 = shapely.Point(18.498214746794215, 17.99761966239229)

        known = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.LineString((self.p6, _p2, _p1)),
                    shapely.LineString((self.p1, _p3, _p1)),
                    shapely.LineString((self.p8, _p4, _p1)),
                    shapely.LineString((self.p7, _p5, _p1)),
                ],
                "_status": ["changed"] * 4,
            }
        )
        observed = neatnet.nodes.consolidate_nodes(
            self.lines_gframe, tolerance=5, preserve_ends=True
        )

        assert_series_equal(known._status, observed._status)
        pytest.geom_test(known, observed, tolerance=0.000001)

    def test_t7(self):
        _p1 = shapely.Point(16.8, 16.3)
        _p2 = shapely.Point(12.40158826840626, 17.838570558434366)
        _p3 = shapely.Point(12.525126265847085, 12.525126265847085)
        _p4 = shapely.Point(18.79856626850639, 12.00238955248935)

        known = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.LineString((self.p6, _p2, _p1)),
                    shapely.LineString((self.p1, _p3, _p1)),
                    shapely.LineString((self.p8, _p4, _p1)),
                ],
                "_status": ["changed"] * 3,
            }
        )
        observed = neatnet.nodes.consolidate_nodes(self.lines_gframe, tolerance=7)

        assert_series_equal(known._status, observed._status)
        pytest.geom_test(known, observed, tolerance=0.000001)

    def test_t7_pe(self):
        _p1 = shapely.Point(16, 15.375)
        _p2 = shapely.Point(12.40158826840626, 17.838570558434366)
        _p3 = shapely.Point(12.525126265847085, 12.525126265847085)
        _p4 = shapely.Point(18.79856626850639, 12.00238955248935)
        _p5 = shapely.Point(19.097500645511904, 18.796667527349204)

        known = geopandas.GeoDataFrame(
            {
                "geometry": [
                    shapely.LineString((self.p6, _p2, _p1)),
                    shapely.LineString((self.p1, _p3, _p1)),
                    shapely.LineString((self.p8, _p4, _p1)),
                    shapely.LineString((self.p7, _p5, _p1)),
                ],
                "_status": ["changed"] * 4,
            }
        )
        observed = neatnet.nodes.consolidate_nodes(
            self.lines_gframe, tolerance=7, preserve_ends=True
        )

        assert_series_equal(known._status, observed._status)
        pytest.geom_test(known, observed, tolerance=0.000001)


def test_fill_attrs():
    known = pandas.DataFrame(
        {
            "attr1": ["a", "b"],
            "attr2": [5, 5],
            "attr3": ["W", "W"],
            "attr4": [["i", "o", 6], ["i", "o", 6]],
        }
    )

    observed = neatnet.nodes._fill_attrs(
        pandas.DataFrame({"attr1": ["a", "b"]}),
        pandas.Series([5, "W", ["i", "o", 6]], index=["attr2", "attr3", "attr4"]),
    )

    assert_frame_equal(known, observed)
