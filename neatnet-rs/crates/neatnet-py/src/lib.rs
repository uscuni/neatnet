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
use geos::Geom;
use geo_traits::to_geo::ToGeoLineString;
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

/// Convert a PyGeoArray (LineString) to Vec<geos::Geometry>.
fn geoarrow_to_geos(input: &PyGeoArray) -> PyResult<Vec<geos::Geometry>> {
    let geo_array = input.inner();

    let ls_array = geo_array.as_line_string_opt().ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err("Expected a GeoArrow LineString array")
    })?;

    let mut geos_geoms = Vec::with_capacity(ls_array.len());
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
        let geos_geom: geos::Geometry = (&gt_ls).try_into().map_err(|e: geos::Error| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "GEOS conversion failed at index {i}: {e}"
            ))
        })?;
        geos_geoms.push(geos_geom);
    }

    Ok(geos_geoms)
}

/// Convert Vec<geos::Geometry> to a PyGeoArray (LineString).
fn geos_to_geoarrow(geoms: &[geos::Geometry]) -> PyResult<PyGeoArray> {
    let gt_linestrings: Vec<geo_types::LineString<f64>> = geoms
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let gt_geom: geo_types::Geometry<f64> =
                g.try_into().map_err(|e: geos::Error| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "GEOS->geo_types conversion failed at index {i}: {e}"
                    ))
                })?;

            match gt_geom {
                geo_types::Geometry::LineString(ls) => Ok(ls),
                other => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Expected LineString at index {i}, got {:?}",
                    other
                ))),
            }
        })
        .collect::<PyResult<Vec<_>>>()?;

    let ls_type = LineStringType::new(Dimension::XY, Arc::new(Metadata::default()));
    let builder = LineStringBuilder::from_line_strings(&gt_linestrings, ls_type);
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
    let geos_geoms = geoarrow_to_geos(&geometries)?;
    let statuses = vec![neatnet_core::EdgeStatus::Original; geos_geoms.len()];

    let mut network = neatnet_core::StreetNetwork {
        geometries: geos_geoms,
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

    let result_geom = geos_to_geoarrow(&network.geometries)?;
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
    let geos_geoms = geoarrow_to_geos(&geometries)?;
    let result = neatnet_core::continuity::coins(&geos_geoms, angle_threshold);

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
    let geometries: Vec<geos::Geometry> = wkt_geometries
        .iter()
        .filter_map(|wkt| geos::Geometry::new_from_wkt(wkt).ok())
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
        .filter_map(|g| g.to_wkt().ok())
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
    let geometries: Vec<geos::Geometry> = wkt_geometries
        .iter()
        .filter_map(|wkt| geos::Geometry::new_from_wkt(wkt).ok())
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
    let lines: Vec<geos::Geometry> = wkt_lines
        .iter()
        .filter_map(|wkt| geos::Geometry::new_from_wkt(wkt).ok())
        .collect();

    let poly = wkt_poly.and_then(|wkt| geos::Geometry::new_from_wkt(&wkt).ok());
    let snap_geoms: Option<Vec<geos::Geometry>> = wkt_snap_to.map(|wkts| {
        wkts.iter()
            .filter_map(|wkt| geos::Geometry::new_from_wkt(wkt).ok())
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
        .filter_map(|g| g.to_wkt().ok())
        .collect();
    let split_wkts: Vec<String> = splitters
        .iter()
        .filter_map(|g| g.to_wkt().ok())
        .collect();
    dict.set_item("edgelines", edge_wkts)?;
    dict.set_item("splitters", split_wkts)?;
    Ok(dict.into())
}
