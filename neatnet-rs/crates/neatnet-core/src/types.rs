//! Core data types for neatnet street network simplification.

use std::fmt;

use geos::Geometry as GGeometry;

/// Tracking status of edges through the simplification pipeline.
///
/// Mirrors the Python `_status` column semantics:
/// - `Original` – edge unchanged from input
/// - `New`      – edge created by the algorithm
/// - `Changed`  – edge modified (split, merged, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeStatus {
    Original,
    New,
    Changed,
}

impl fmt::Display for EdgeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeStatus::Original => write!(f, "original"),
            EdgeStatus::New => write!(f, "new"),
            EdgeStatus::Changed => write!(f, "changed"),
        }
    }
}

impl EdgeStatus {
    /// Aggregate a group of statuses (mirrors Python `_status()` callable).
    ///
    /// Rules:
    /// - If all are `New` → `New`
    /// - If exactly one element → that element
    /// - Otherwise → `Changed`
    pub fn aggregate(statuses: &[EdgeStatus]) -> EdgeStatus {
        if statuses.len() == 1 {
            return statuses[0];
        }
        if statuses.iter().all(|s| *s == EdgeStatus::New) {
            EdgeStatus::New
        } else {
            EdgeStatus::Changed
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeStatus::Original => "original",
            EdgeStatus::New => "new",
            EdgeStatus::Changed => "changed",
        }
    }

    pub fn from_str(s: &str) -> Option<EdgeStatus> {
        match s {
            "original" => Some(EdgeStatus::Original),
            "new" => Some(EdgeStatus::New),
            "changed" => Some(EdgeStatus::Changed),
            _ => None,
        }
    }
}

/// A street network: geometry stored as GEOS LineStrings, attributes in Arrow.
///
/// This is the main data structure flowing through the pipeline.
/// Geometry and attributes are stored in parallel arrays (same length, same order).
pub struct StreetNetwork {
    /// LineString geometries (one per edge).
    pub geometries: Vec<GGeometry>,
    /// Per-edge status tracking.
    pub statuses: Vec<EdgeStatus>,
    /// Optional Arrow RecordBatch holding non-geometry attribute columns.
    pub attributes: Option<arrow::record_batch::RecordBatch>,
    /// Coordinate reference system identifier (e.g. "EPSG:32637").
    pub crs: Option<String>,
}

impl StreetNetwork {
    /// Number of edges.
    pub fn len(&self) -> usize {
        self.geometries.len()
    }

    /// Whether the network has no edges.
    pub fn is_empty(&self) -> bool {
        self.geometries.is_empty()
    }
}

/// Face artifacts detected by the polygonization & FAI pipeline.
pub struct Artifacts {
    /// Polygon geometries (one per artifact face).
    pub geometries: Vec<GGeometry>,
    /// Number of network nodes touching each artifact.
    pub node_count: Vec<usize>,
    /// Face artifact index value per polygon.
    pub face_artifact_index: Vec<f64>,
    /// Whether each polygon is classified as an artifact.
    pub is_artifact: Vec<bool>,
    /// Connected-component label from contiguity graph.
    pub component_labels: Vec<usize>,
    /// CES classification counts per artifact.
    pub c_count: Vec<usize>,
    pub e_count: Vec<usize>,
    pub s_count: Vec<usize>,
    /// Total stroke count per artifact.
    pub stroke_count: Vec<usize>,
}

/// Parameters controlling the neatify pipeline.
///
/// Mirrors all keyword arguments of Python `neatify()`.
#[derive(Debug, Clone)]
pub struct NeatifyParams {
    pub max_segment_length: f64,
    pub min_dangle_length: f64,
    pub clip_limit: f64,
    pub simplification_factor: f64,
    pub consolidation_tolerance: f64,
    pub artifact_threshold: Option<f64>,
    pub artifact_threshold_fallback: f64,
    pub area_threshold_blocks: f64,
    pub isoareal_threshold_blocks: f64,
    pub area_threshold_circles: f64,
    pub isoareal_threshold_circles_enclosed: f64,
    pub isoperimetric_threshold_circles_touching: f64,
    pub angle_threshold: f64,
    pub eps: f64,
    pub n_loops: usize,
}

impl Default for NeatifyParams {
    fn default() -> Self {
        Self {
            max_segment_length: 1.0,
            min_dangle_length: 20.0,
            clip_limit: 2.0,
            simplification_factor: 2.0,
            consolidation_tolerance: 10.0,
            artifact_threshold: None,
            artifact_threshold_fallback: 7.0,
            area_threshold_blocks: 1e5,
            isoareal_threshold_blocks: 0.5,
            area_threshold_circles: 5e4,
            isoareal_threshold_circles_enclosed: 0.75,
            isoperimetric_threshold_circles_touching: 0.9,
            angle_threshold: 120.0,
            eps: 1e-4,
            n_loops: 2,
        }
    }
}

/// Result of the COINS continuity analysis for a single edge.
#[derive(Debug, Clone)]
pub struct CoinsInfo {
    /// Stroke group ID.
    pub group: usize,
    /// Whether this edge is at the end of its stroke.
    pub is_end: bool,
    /// Total length of the stroke group.
    pub stroke_length: f64,
    /// Number of edges in the stroke group.
    pub stroke_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_status_aggregate() {
        assert_eq!(
            EdgeStatus::aggregate(&[EdgeStatus::New]),
            EdgeStatus::New
        );
        assert_eq!(
            EdgeStatus::aggregate(&[EdgeStatus::Original]),
            EdgeStatus::Original
        );
        assert_eq!(
            EdgeStatus::aggregate(&[EdgeStatus::New, EdgeStatus::New]),
            EdgeStatus::New
        );
        assert_eq!(
            EdgeStatus::aggregate(&[EdgeStatus::New, EdgeStatus::Original]),
            EdgeStatus::Changed
        );
        assert_eq!(
            EdgeStatus::aggregate(&[EdgeStatus::Original, EdgeStatus::Changed]),
            EdgeStatus::Changed
        );
    }

    #[test]
    fn test_edge_status_display() {
        assert_eq!(EdgeStatus::Original.to_string(), "original");
        assert_eq!(EdgeStatus::New.to_string(), "new");
        assert_eq!(EdgeStatus::Changed.to_string(), "changed");
    }

    #[test]
    fn test_edge_status_roundtrip() {
        for status in [EdgeStatus::Original, EdgeStatus::New, EdgeStatus::Changed] {
            assert_eq!(EdgeStatus::from_str(status.as_str()), Some(status));
        }
        assert_eq!(EdgeStatus::from_str("invalid"), None);
    }

    #[test]
    fn test_neatify_params_default() {
        let p = NeatifyParams::default();
        assert_eq!(p.max_segment_length, 1.0);
        assert_eq!(p.n_loops, 2);
        assert!(p.artifact_threshold.is_none());
    }
}
