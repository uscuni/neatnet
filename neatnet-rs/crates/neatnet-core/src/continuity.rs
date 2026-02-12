//! COINS (Continuity in Street Networks) algorithm.
//!
//! Ports the `momepy.COINS` implementation: given a set of LineString
//! geometries and an angle threshold, assigns each edge to a "stroke"
//! group by greedily pairing edges that share an endpoint with the
//! smallest deflection angle, provided it is below the threshold.

use std::collections::HashMap;
use std::f64::consts::PI;

use geos::{Geom, Geometry as GGeometry};
use petgraph::unionfind::UnionFind;

/// Result of COINS analysis for a collection of edges.
#[derive(Debug, Clone)]
pub struct CoinsResult {
    /// Stroke group ID per input edge (0-based).
    pub group: Vec<usize>,
    /// Whether each edge is at the end of its stroke.
    pub is_end: Vec<bool>,
    /// Total length of each edge's stroke group.
    pub stroke_length: Vec<f64>,
    /// Number of edges in each edge's stroke group.
    pub stroke_count: Vec<usize>,
}

/// Run the COINS algorithm on a set of LineString geometries.
///
/// # Arguments
/// * `geometries` – slice of GEOS LineString geometries
/// * `angle_threshold` – maximum deflection angle (degrees) for pairing (default 120°)
///
/// # Returns
/// A `CoinsResult` with per-edge stroke assignments.
pub fn coins(geometries: &[GGeometry], angle_threshold: f64) -> CoinsResult {
    let n = geometries.len();
    if n == 0 {
        return CoinsResult {
            group: vec![],
            is_end: vec![],
            stroke_length: vec![],
            stroke_count: vec![],
        };
    }

    // 1. Extract endpoints for each edge and optionally split at
    //    intermediate coordinates (flow_mode=True equivalent).
    //    For each edge, store pairs of (start_coord, end_coord) for
    //    each two-point segment.
    let segments = extract_segments(geometries);

    // 2. Build an endpoint→segment adjacency map.
    //    Key: rounded coordinate (to handle floating point), Value: list of segment indices
    let mut endpoint_map: HashMap<CoordKey, Vec<usize>> = HashMap::new();
    for (seg_idx, seg) in segments.iter().enumerate() {
        let start_key = coord_key(seg.start);
        let end_key = coord_key(seg.end);
        endpoint_map.entry(start_key).or_default().push(seg_idx);
        endpoint_map.entry(end_key).or_default().push(seg_idx);
    }

    // 3. Greedy best-link pairing: at each shared endpoint, pair segments
    //    with the smallest deflection angle (if below threshold).
    let mut uf = UnionFind::new(segments.len());
    let threshold_rad = angle_threshold * PI / 180.0;

    for (_coord, seg_indices) in &endpoint_map {
        if seg_indices.len() < 2 {
            continue;
        }
        // For each segment at this node, find its best partner
        let mut best_pairs: Vec<(usize, usize, f64)> = Vec::new();

        for i in 0..seg_indices.len() {
            let mut best_angle = f64::MAX;
            let mut best_partner = None;

            for j in 0..seg_indices.len() {
                if i == j {
                    continue;
                }
                let si = seg_indices[i];
                let sj = seg_indices[j];
                let angle = deflection_angle(&segments[si], &segments[sj], _coord);
                if angle < best_angle {
                    best_angle = angle;
                    best_partner = Some(j);
                }
            }

            if let Some(partner) = best_partner {
                if best_angle < threshold_rad {
                    let si = seg_indices[i];
                    let sj = seg_indices[partner];
                    best_pairs.push((si, sj, best_angle));
                }
            }
        }

        // Only pair if both segments agree on each other as best partner
        for &(si, sj, _) in &best_pairs {
            // Check reciprocity: sj's best partner at this node should be si
            let reciprocal = best_pairs
                .iter()
                .any(|&(a, b, _)| a == sj && b == si);
            if reciprocal {
                uf.union(si, sj);
            }
        }
    }

    // 4. Map segment groups back to original edge indices.
    //    Each original edge may have been split into multiple segments;
    //    we need the group of the *original edge* (union of its segments' groups).
    let seg_labels: Vec<usize> = (0..segments.len()).map(|i| uf.find(i)).collect();

    // Build segment_group -> canonical group label
    let mut group_remap: HashMap<usize, usize> = HashMap::new();
    let mut next_group = 0usize;
    let mut edge_groups = vec![0usize; n];

    for edge_idx in 0..n {
        // Find the segment(s) belonging to this edge
        // and use the first segment's group as the edge group
        let first_seg = edge_segment_start(edge_idx, geometries);
        let raw_group = seg_labels[first_seg];
        let group = *group_remap.entry(raw_group).or_insert_with(|| {
            let g = next_group;
            next_group += 1;
            g
        });
        edge_groups[edge_idx] = group;
    }

    // 5. Compute per-group aggregates.
    let mut group_lengths: HashMap<usize, f64> = HashMap::new();
    let mut group_counts: HashMap<usize, usize> = HashMap::new();
    for (edge_idx, &group) in edge_groups.iter().enumerate() {
        let len = geometry_length(geometries, edge_idx);
        *group_lengths.entry(group).or_default() += len;
        *group_counts.entry(group).or_default() += 1;
    }

    // 6. Determine which edges are stroke-ends.
    //    An edge is a stroke-end if one of its endpoints connects to zero
    //    other edges in the same group, or the edge is a singleton stroke.
    let is_end = compute_stroke_ends(&edge_groups, geometries, &endpoint_map, &segments);

    let stroke_length: Vec<f64> = edge_groups
        .iter()
        .map(|g| *group_lengths.get(g).unwrap_or(&0.0))
        .collect();
    let stroke_count: Vec<usize> = edge_groups
        .iter()
        .map(|g| *group_counts.get(g).unwrap_or(&0))
        .collect();

    CoinsResult {
        group: edge_groups,
        is_end,
        stroke_length,
        stroke_count,
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// A two-point segment extracted from an edge.
#[derive(Debug, Clone)]
struct Segment {
    start: [f64; 2],
    end: [f64; 2],
    /// Index of the original edge this segment belongs to.
    edge_idx: usize,
}

/// Coordinate key for hashing (rounded to avoid float issues).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CoordKey {
    x: i64,
    y: i64,
}

fn coord_key(c: [f64; 2]) -> CoordKey {
    // Round to ~1e-8 precision
    CoordKey {
        x: (c[0] * 1e8).round() as i64,
        y: (c[1] * 1e8).round() as i64,
    }
}

/// Extract two-point segments from all edges (flow_mode=True).
/// Each edge with N coordinates produces N-1 segments.
fn extract_segments(geometries: &[GGeometry]) -> Vec<Segment> {
    let mut segments = Vec::new();
    for (edge_idx, geom) in geometries.iter().enumerate() {
        let Ok(cs) = geom.get_coord_seq() else {
            continue;
        };
        let Ok(n) = cs.size() else { continue };
        if n < 2 {
            continue;
        }
        for i in 0..n - 1 {
            let (Ok(x0), Ok(y0)) = (cs.get_x(i), cs.get_y(i)) else {
                continue;
            };
            let (Ok(x1), Ok(y1)) = (cs.get_x(i + 1), cs.get_y(i + 1)) else {
                continue;
            };
            segments.push(Segment {
                start: [x0, y0],
                end: [x1, y1],
                edge_idx,
            });
        }
    }
    segments
}

/// Get the global segment index for the first segment of an edge.
fn edge_segment_start(edge_idx: usize, geometries: &[GGeometry]) -> usize {
    let mut offset = 0;
    for (i, geom) in geometries.iter().enumerate() {
        if i == edge_idx {
            return offset;
        }
        let n = geom
            .get_coord_seq()
            .and_then(|cs| cs.size())
            .unwrap_or(0);
        offset += if n > 1 { n - 1 } else { 0 };
    }
    offset
}

/// Compute deflection angle between two segments meeting at `node`.
/// Returns angle in radians in [0, π].
fn deflection_angle(s1: &Segment, s2: &Segment, node: &CoordKey) -> f64 {
    let k1 = coord_key(s1.start);
    let k2 = coord_key(s1.end);
    let k3 = coord_key(s2.start);
    let k4 = coord_key(s2.end);

    // Determine the "away" direction from the shared node for each segment
    let (dx1, dy1) = if k2 == *node {
        (s1.start[0] - s1.end[0], s1.start[1] - s1.end[1])
    } else if k1 == *node {
        (s1.end[0] - s1.start[0], s1.end[1] - s1.start[1])
    } else {
        return f64::MAX;
    };

    let (dx2, dy2) = if k3 == *node {
        (s2.end[0] - s2.start[0], s2.end[1] - s2.start[1])
    } else if k4 == *node {
        (s2.start[0] - s2.end[0], s2.start[1] - s2.end[1])
    } else {
        return f64::MAX;
    };

    // Angle between the two direction vectors
    let dot = dx1 * dx2 + dy1 * dy2;
    let mag1 = (dx1 * dx1 + dy1 * dy1).sqrt();
    let mag2 = (dx2 * dx2 + dy2 * dy2).sqrt();

    if mag1 < 1e-12 || mag2 < 1e-12 {
        return f64::MAX;
    }

    let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    // Deflection = π - angle (we want the deviation from straight)
    PI - cos_angle.acos()
}

/// Get the length of a geometry.
fn geometry_length(geometries: &[GGeometry], idx: usize) -> f64 {
    geometries[idx].length().unwrap_or(0.0)
}

/// Determine which edges are stroke-ends.
fn compute_stroke_ends(
    edge_groups: &[usize],
    geometries: &[GGeometry],
    _endpoint_map: &HashMap<CoordKey, Vec<usize>>,
    _segments: &[Segment],
) -> Vec<bool> {
    let n = edge_groups.len();
    let mut is_end = vec![false; n];

    // Build edge endpoint coordinates
    for edge_idx in 0..n {
        let Ok(cs) = geometries[edge_idx].get_coord_seq() else {
            is_end[edge_idx] = true;
            continue;
        };
        let Ok(npts) = cs.size() else {
            is_end[edge_idx] = true;
            continue;
        };
        if npts < 2 {
            is_end[edge_idx] = true;
            continue;
        }

        let group = edge_groups[edge_idx];

        // Check start endpoint
        let (Ok(sx), Ok(sy)) = (cs.get_x(0), cs.get_y(0)) else {
            is_end[edge_idx] = true;
            continue;
        };
        let start_key = coord_key([sx, sy]);

        // Check end endpoint
        let (Ok(ex), Ok(ey)) = (cs.get_x(npts - 1), cs.get_y(npts - 1)) else {
            is_end[edge_idx] = true;
            continue;
        };
        let end_key = coord_key([ex, ey]);

        // An edge is a stroke-end if at either endpoint there is no other
        // edge from the same group
        let start_same_group = _endpoint_map
            .get(&start_key)
            .map(|segs| {
                segs.iter()
                    .any(|&si| _segments[si].edge_idx != edge_idx && edge_groups.get(_segments[si].edge_idx) == Some(&group))
            })
            .unwrap_or(false);

        let end_same_group = _endpoint_map
            .get(&end_key)
            .map(|segs| {
                segs.iter()
                    .any(|&si| _segments[si].edge_idx != edge_idx && edge_groups.get(_segments[si].edge_idx) == Some(&group))
            })
            .unwrap_or(false);

        if !start_same_group || !end_same_group {
            is_end[edge_idx] = true;
        }
    }

    is_end
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line(coords: &[[f64; 2]]) -> GGeometry {
        let wkt = format!(
            "LINESTRING ({})",
            coords
                .iter()
                .map(|c| format!("{} {}", c[0], c[1]))
                .collect::<Vec<_>>()
                .join(", ")
        );
        GGeometry::new_from_wkt(&wkt).unwrap()
    }

    #[test]
    fn test_coins_empty() {
        let result = coins(&[], 120.0);
        assert!(result.group.is_empty());
    }

    #[test]
    fn test_coins_single_edge() {
        let geom = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let result = coins(&[geom], 120.0);
        assert_eq!(result.group.len(), 1);
        assert_eq!(result.stroke_count, vec![1]);
        assert!(result.is_end[0]);
    }

    #[test]
    fn test_coins_straight_pair() {
        // Two collinear segments should be grouped together
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let result = coins(&[g1, g2], 120.0);
        assert_eq!(result.group[0], result.group[1]);
        assert_eq!(result.stroke_count[0], 2);
    }

    #[test]
    fn test_coins_perpendicular() {
        // Two perpendicular segments should NOT be grouped (90° deflection > threshold
        // only if threshold is small enough)
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let result = coins(&[g1, g2], 30.0); // strict threshold
        assert_ne!(result.group[0], result.group[1]);
    }
}
