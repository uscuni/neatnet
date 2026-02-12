//! PyO3 Python bindings for neatnet-rs.
//!
//! Exposes the Rust neatify pipeline to Python via GeoArrow for
//! zero-copy geometry transfer and Arrow for attribute columns.

use geos::Geom;
use pyo3::prelude::*;

/// The neatnet_rs Python module.
///
/// Provides `neatify()` – the main entry point for street network
/// simplification – plus lower-level functions for testing.
#[pymodule]
fn _neatnet_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(neatify, m)?)?;
    m.add_function(wrap_pyfunction!(coins, m)?)?;
    m.add_function(wrap_pyfunction!(voronoi_skeleton, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}

/// Return the version of the neatnet-rs Rust core.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Run COINS continuity analysis on a list of WKT geometries.
///
/// This is primarily for testing/benchmarking the COINS implementation.
///
/// Parameters
/// ----------
/// wkt_geometries : list[str]
///     WKT representations of LineString geometries.
/// angle_threshold : float, default 120.0
///     Maximum deflection angle for stroke pairing.
///
/// Returns
/// -------
/// dict
///     Dictionary with keys 'group', 'is_end', 'stroke_length', 'stroke_count'.
#[pyfunction]
#[pyo3(signature = (wkt_geometries, angle_threshold=120.0))]
fn coins(
    wkt_geometries: Vec<String>,
    angle_threshold: f64,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
    // Parse WKT geometries
    let geometries: Vec<geos::Geometry> = wkt_geometries
        .iter()
        .filter_map(|wkt| geos::Geometry::new_from_wkt(wkt).ok())
        .collect();

    let result = neatnet_core::continuity::coins(&geometries, angle_threshold);

    Python::with_gil(|py| {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("group", &result.group)?;
        dict.set_item("is_end", &result.is_end)?;
        dict.set_item("stroke_length", &result.stroke_length)?;
        dict.set_item("stroke_count", &result.stroke_count)?;
        dict.set_item("n_segments", result.n_segments)?;
        dict.set_item("n_p1_confirmed", result.n_p1_confirmed)?;
        dict.set_item("n_p2_confirmed", result.n_p2_confirmed)?;
        Ok(dict.into())
    })
}

/// Compute voronoi skeleton from lines within a polygon.
///
/// Parameters
/// ----------
/// wkt_lines : list[str]
///     WKT representations of LineString geometries.
/// wkt_poly : str or None
///     WKT representation of the bounding polygon.
/// wkt_snap_to : list[str] or None
///     WKT geometries to snap skeleton endpoints to.
/// max_segment_length : float, default 1.0
/// clip_limit : float, default 2.0
///
/// Returns
/// -------
/// dict
///     Dictionary with 'edgelines' and 'splitters' as lists of WKT strings.
#[pyfunction]
#[pyo3(signature = (wkt_lines, wkt_poly=None, wkt_snap_to=None, max_segment_length=1.0, clip_limit=2.0))]
fn voronoi_skeleton(
    wkt_lines: Vec<String>,
    wkt_poly: Option<String>,
    wkt_snap_to: Option<Vec<String>>,
    max_segment_length: f64,
    clip_limit: f64,
) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
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

    Python::with_gil(|py| {
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
    })
}

/// Simplify a street network.
///
/// This is the main entry point. It accepts a GeoDataFrame-like input
/// (via GeoArrow/Arrow FFI) and returns a simplified version.
///
/// Parameters
/// ----------
/// wkt_geometries : list[str]
///     WKT representations of LineString geometries.
///     (Temporary interface -- will be replaced with GeoArrow FFI.)
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
/// list[str]
///     WKT representations of the simplified geometries.
///     (Temporary interface -- will be replaced with GeoArrow FFI.)
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
fn neatify(
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
) -> PyResult<Vec<String>> {
    // Parse WKT geometries
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

    // Convert back to WKT
    let result: Vec<String> = network
        .geometries
        .iter()
        .filter_map(|g| g.to_wkt().ok())
        .collect();

    Ok(result)
}
