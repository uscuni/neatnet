//! PyO3 Python bindings for neatnet-rs.
//!
//! Exposes the Rust neatify pipeline to Python via GeoArrow for
//! zero-copy geometry transfer and Arrow for attribute columns.

use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use geoarrow::array::{
    AsGeoArrowArray, GeoArrowArray, GeoArrowArrayAccessor, LineStringBuilder,
};
use geoarrow::datatypes::{Dimension, LineStringType, Metadata};
use geo_traits::to_geo::ToGeoLineString;
use geo_types::LineString;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use pyo3_geoarrow::PyGeoArray;

/// The neatnet_rs Python module.
#[pymodule]
fn _neatnet_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neatify, m)?)?;
    m.add_function(wrap_pyfunction!(neatify_wkt, m)?)?;
    m.add_function(wrap_pyfunction!(coins, m)?)?;
    m.add_function(wrap_pyfunction!(coins_wkt, m)?)?;
    m.add_function(wrap_pyfunction!(voronoi_skeleton_wkt, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

/// Return the version of the neatnet-rs Rust core.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert a PyGeoArray (LineString) to Vec<geo_types::LineString<f64>>.
fn geoarrow_to_geo(input: &PyGeoArray) -> PyResult<Vec<LineString<f64>>> {
    let geo_array = input.inner();

    let ls_array = geo_array.as_line_string_opt().ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err("Expected a GeoArrow LineString array")
    })?;

    let mut geoms = Vec::with_capacity(ls_array.len());
    for i in 0..ls_array.len() {
        if ls_array.is_null(i) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Null geometry at index {i}"
            )));
        }
        let scalar = ls_array.value(i).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Array access error at index {i}: {e}"
            ))
        })?;
        let gt_ls: geo_types::LineString<f64> = scalar.to_line_string();
        geoms.push(gt_ls);
    }

    Ok(geoms)
}

/// Convert Vec<geo_types::LineString<f64>> to a PyGeoArray (LineString).
fn geo_to_geoarrow(geoms: &[LineString<f64>]) -> PyResult<PyGeoArray> {
    let ls_type = LineStringType::new(Dimension::XY, Arc::new(Metadata::default()));
    let builder = LineStringBuilder::from_line_strings(geoms, ls_type);
    let ls_array = builder.finish();
    let geo_array: Arc<dyn GeoArrowArray> = Arc::new(ls_array);
    Ok(PyGeoArray::new(geo_array))
}

/// Convert Vec<EdgeStatus> to a PyArrow StringArray via PyCapsule FFI.
fn statuses_to_arrow(
    py: Python<'_>,
    statuses: &[neatnet_core::EdgeStatus],
) -> PyResult<Py<PyAny>> {
    let str_array: StringArray = statuses.iter().map(|s| Some(s.as_str())).collect();
    let array_ref: ArrayRef = Arc::new(str_array);
    let field = arrow::datatypes::Field::new("status", arrow::datatypes::DataType::Utf8, false);
    let py_array = PyArray::new(array_ref, Arc::new(field));
    py_array.to_pyarrow(py).map(|bound| bound.unbind())
}

/// Parse a WKT string to a geo_types::LineString<f64>.
fn wkt_to_linestring(wkt_str: &str) -> Option<LineString<f64>> {
    use std::str::FromStr;
    let wkt_obj = wkt::Wkt::from_str(wkt_str).ok()?;
    let geom: geo_types::Geometry<f64> = wkt_obj.try_into().ok()?;
    match geom {
        geo_types::Geometry::LineString(ls) => Some(ls),
        _ => None,
    }
}

/// Convert a geo_types::LineString<f64> to a WKT string.
fn linestring_to_wkt(ls: &LineString<f64>) -> String {
    use std::fmt::Write;
    let mut s = String::from("LINESTRING (");
    for (i, coord) in ls.0.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        write!(s, "{} {}", coord.x, coord.y).unwrap();
    }
    s.push(')');
    s
}

/// Parse a WKT string to a geo_types::Polygon<f64>.
fn wkt_to_polygon(wkt_str: &str) -> Option<geo_types::Polygon<f64>> {
    use std::str::FromStr;
    let wkt_obj = wkt::Wkt::from_str(wkt_str).ok()?;
    let geom: geo_types::Geometry<f64> = wkt_obj.try_into().ok()?;
    match geom {
        geo_types::Geometry::Polygon(p) => Some(p),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// GeoArrow-based functions (primary API)
// ---------------------------------------------------------------------------

/// Simplify a street network.
///
/// Accepts a GeoArrow LineString array and returns simplified geometries
/// plus an Arrow StringArray of edge statuses.
///
/// Parameters
/// ----------
/// geometries : GeoArrow LineString array
///     Input street network geometries (via Arrow PyCapsule Interface).
///     Must use native GeoArrow encoding (not WKB). From geopandas, use:
///     ``gdf.geometry.to_arrow(geometry_encoding="geoarrow")``
/// max_segment_length : float, default 1.0
/// min_dangle_length : float, default 20.0
/// clip_limit : float, default 2.0
/// simplification_factor : float, default 2.0
/// consolidation_tolerance : float, default 10.0
/// artifact_threshold : float or None, default None
/// artifact_threshold_fallback : float, default 7.0
/// angle_threshold : float, default 120.0
/// eps : float, default 1e-4
/// n_loops : int, default 2
///
/// Returns
/// -------
/// tuple[GeoArrow LineString array, Arrow StringArray]
///     (simplified_geometries, status_array)
#[pyfunction]
#[pyo3(signature = (
    geometries,
    max_segment_length=1.0,
    min_dangle_length=20.0,
    clip_limit=2.0,
    simplification_factor=2.0,
    consolidation_tolerance=10.0,
    artifact_threshold=None,
    artifact_threshold_fallback=7.0,
    angle_threshold=120.0,
    eps=1e-4,
    n_loops=2,
))]
fn neatify(
    py: Python<'_>,
    geometries: PyGeoArray,
    max_segment_length: f64,
    min_dangle_length: f64,
    clip_limit: f64,
    simplification_factor: f64,
    consolidation_tolerance: f64,
    artifact_threshold: Option<f64>,
    artifact_threshold_fallback: f64,
    angle_threshold: f64,
    eps: f64,
    n_loops: usize,
) -> PyResult<(PyGeoArray, Py<PyAny>)> {
    let geo_geoms = geoarrow_to_geo(&geometries)?;
    let statuses = vec![neatnet_core::EdgeStatus::Original; geo_geoms.len()];

    let mut network = neatnet_core::StreetNetwork {
        geometries: geo_geoms,
        statuses,
        attributes: None,
        crs: None,
    };

    let params = neatnet_core::NeatifyParams {
        max_segment_length,
        min_dangle_length,
        clip_limit,
        simplification_factor,
        consolidation_tolerance,
        artifact_threshold,
        artifact_threshold_fallback,
        angle_threshold,
        eps,
        n_loops,
        ..Default::default()
    };

    neatnet_core::neatify(&mut network, &params, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let result_geom = geo_to_geoarrow(&network.geometries)?;
    let result_status = statuses_to_arrow(py, &network.statuses)?;

    Ok((result_geom, result_status))
}

/// Run COINS continuity analysis on a GeoArrow LineString array.
///
/// Parameters
/// ----------
/// geometries : GeoArrow LineString array
///     Input geometries (via Arrow PyCapsule Interface).
///     Must use native GeoArrow encoding (not WKB).
/// angle_threshold : float, default 120.0
///
/// Returns
/// -------
/// dict
///     Dictionary with keys 'group', 'is_end', 'stroke_length', 'stroke_count'.
#[pyfunction]
#[pyo3(signature = (geometries, angle_threshold=120.0))]
fn coins(
    py: Python<'_>,
    geometries: PyGeoArray,
    angle_threshold: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let geo_geoms = geoarrow_to_geo(&geometries)?;
    let result = neatnet_core::continuity::coins(&geo_geoms, angle_threshold);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("group", &result.group)?;
    dict.set_item("is_end", &result.is_end)?;
    dict.set_item("stroke_length", &result.stroke_length)?;
    dict.set_item("stroke_count", &result.stroke_count)?;
    dict.set_item("n_segments", result.n_segments)?;
    dict.set_item("n_p1_confirmed", result.n_p1_confirmed)?;
    dict.set_item("n_p2_confirmed", result.n_p2_confirmed)?;
    Ok(dict.into())
}

// ---------------------------------------------------------------------------
// WKT-based functions (for testing and backwards compatibility)
// ---------------------------------------------------------------------------

/// Simplify a street network (WKT interface).
#[pyfunction]
#[pyo3(signature = (
    wkt_geometries,
    max_segment_length=1.0,
    min_dangle_length=20.0,
    clip_limit=2.0,
    simplification_factor=2.0,
    consolidation_tolerance=10.0,
    artifact_threshold=None,
    artifact_threshold_fallback=7.0,
    angle_threshold=120.0,
    eps=1e-4,
    n_loops=2,
))]
fn neatify_wkt(
    py: Python<'_>,
    wkt_geometries: Vec<String>,
    max_segment_length: f64,
    min_dangle_length: f64,
    clip_limit: f64,
    simplification_factor: f64,
    consolidation_tolerance: f64,
    artifact_threshold: Option<f64>,
    artifact_threshold_fallback: f64,
    angle_threshold: f64,
    eps: f64,
    n_loops: usize,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let geometries: Vec<LineString<f64>> = wkt_geometries
        .iter()
        .filter_map(|wkt| wkt_to_linestring(wkt))
        .collect();

    let statuses = vec![neatnet_core::EdgeStatus::Original; geometries.len()];

    let mut network = neatnet_core::StreetNetwork {
        geometries,
        statuses,
        attributes: None,
        crs: None,
    };

    let params = neatnet_core::NeatifyParams {
        max_segment_length,
        min_dangle_length,
        clip_limit,
        simplification_factor,
        consolidation_tolerance,
        artifact_threshold,
        artifact_threshold_fallback,
        angle_threshold,
        eps,
        n_loops,
        ..Default::default()
    };

    neatnet_core::neatify(&mut network, &params, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

    let geom_wkts: Vec<String> = network
        .geometries
        .iter()
        .map(|g| linestring_to_wkt(g))
        .collect();
    let status_strs: Vec<&str> = network.statuses.iter().map(|s| s.as_str()).collect();

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("geometries", geom_wkts)?;
    dict.set_item("statuses", status_strs)?;
    Ok(dict.into())
}

/// Run COINS continuity analysis (WKT interface).
#[pyfunction]
#[pyo3(signature = (wkt_geometries, angle_threshold=120.0))]
fn coins_wkt(
    py: Python<'_>,
    wkt_geometries: Vec<String>,
    angle_threshold: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let geometries: Vec<LineString<f64>> = wkt_geometries
        .iter()
        .filter_map(|wkt| wkt_to_linestring(wkt))
        .collect();

    let result = neatnet_core::continuity::coins(&geometries, angle_threshold);

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("group", &result.group)?;
    dict.set_item("is_end", &result.is_end)?;
    dict.set_item("stroke_length", &result.stroke_length)?;
    dict.set_item("stroke_count", &result.stroke_count)?;
    dict.set_item("n_segments", result.n_segments)?;
    dict.set_item("n_p1_confirmed", result.n_p1_confirmed)?;
    dict.set_item("n_p2_confirmed", result.n_p2_confirmed)?;
    Ok(dict.into())
}

/// Compute voronoi skeleton from lines within a polygon (WKT interface).
#[pyfunction]
#[pyo3(signature = (wkt_lines, wkt_poly=None, wkt_snap_to=None, max_segment_length=1.0, clip_limit=2.0))]
fn voronoi_skeleton_wkt(
    py: Python<'_>,
    wkt_lines: Vec<String>,
    wkt_poly: Option<String>,
    wkt_snap_to: Option<Vec<String>>,
    max_segment_length: f64,
    clip_limit: f64,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let lines: Vec<LineString<f64>> = wkt_lines
        .iter()
        .filter_map(|wkt| wkt_to_linestring(wkt))
        .collect();

    let poly = wkt_poly.and_then(|wkt| wkt_to_polygon(&wkt));
    let snap_geoms: Option<Vec<LineString<f64>>> = wkt_snap_to.map(|wkts| {
        wkts.iter()
            .filter_map(|wkt| wkt_to_linestring(wkt))
            .collect()
    });

    let (edgelines, splitters) = neatnet_core::geometry::voronoi_skeleton(
        &lines,
        poly.as_ref(),
        snap_geoms.as_deref(),
        max_segment_length,
        None,
        None,
        clip_limit,
        None,
    );

    let dict = pyo3::types::PyDict::new(py);
    let edge_wkts: Vec<String> = edgelines
        .iter()
        .map(|g| linestring_to_wkt(g))
        .collect();
    let split_wkts: Vec<String> = splitters
        .iter()
        .map(|g| linestring_to_wkt(g))
        .collect();
    dict.set_item("edgelines", edge_wkts)?;
    dict.set_item("splitters", split_wkts)?;
    Ok(dict.into())
}
