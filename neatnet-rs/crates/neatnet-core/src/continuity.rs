//! COINS (Continuity in Street Networks) algorithm.
//!
//! Ports the `momepy.COINS` implementation: given a set of LineString
//! geometries and an angle threshold, assigns each edge to a "stroke"
//! group by greedily pairing edges at shared endpoints with the
//! highest interior angle (most collinear continuation), provided
//! the angle exceeds the threshold.
//!
//! Key design: each segment has independent best-links at its p1 (start)
//! and p2 (end) endpoints. Links are cross-checked for reciprocity, then
//! groups are formed by chain-walking along confirmed links.

use std::collections::HashMap;

use geos::{Geom, Geometry as GGeometry};

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
    /// Debug: number of segments
    pub n_segments: usize,
    /// Debug: number of confirmed p1 links
    pub n_p1_confirmed: usize,
    /// Debug: number of confirmed p2 links
    pub n_p2_confirmed: usize,
}

/// Run the COINS algorithm on a set of LineString geometries.
///
/// # Arguments
/// * `geometries` – slice of GEOS LineString geometries
/// * `angle_threshold` – minimum interior angle (degrees) for pairing
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
            n_segments: 0,
            n_p1_confirmed: 0,
            n_p2_confirmed: 0,
        };
    }

    // 1. Split edges into 2-point segments.
    let segments = extract_segments(geometries);
    let n_segs = segments.len();

    // 2. Build endpoint → segment adjacency.
    //    p1_neighbors[i] = segments sharing the same start point as segment i
    //    p2_neighbors[i] = segments sharing the same end point as segment i
    let mut point_to_segs: HashMap<CoordKey, Vec<usize>> = HashMap::new();
    for (idx, seg) in segments.iter().enumerate() {
        point_to_segs
            .entry(coord_key(seg.start))
            .or_default()
            .push(idx);
        point_to_segs
            .entry(coord_key(seg.end))
            .or_default()
            .push(idx);
    }

    let mut p1_neighbors: Vec<Vec<usize>> = vec![vec![]; n_segs];
    let mut p2_neighbors: Vec<Vec<usize>> = vec![vec![]; n_segs];

    for (idx, seg) in segments.iter().enumerate() {
        let sk = coord_key(seg.start);
        if let Some(others) = point_to_segs.get(&sk) {
            p1_neighbors[idx] = others.iter().copied().filter(|&j| j != idx).collect();
        }
        let ek = coord_key(seg.end);
        if let Some(others) = point_to_segs.get(&ek) {
            p2_neighbors[idx] = others.iter().copied().filter(|&j| j != idx).collect();
        }
    }

    // 3. Compute angle pairs and find best link at each endpoint.
    //    angle_pairs[(i, j)] = interior angle between segments i and j
    let mut angle_pairs: HashMap<(usize, usize), f64> = HashMap::new();

    // best_p1[i] = (best_partner_idx, best_angle) at p1, or None
    let mut best_p1: Vec<Option<(usize, f64)>> = vec![None; n_segs];
    let mut best_p2: Vec<Option<(usize, f64)>> = vec![None; n_segs];

    for edge in 0..n_segs {
        // Best link at p1 (start endpoint)
        let mut p1_best_angle = 0.0_f64;
        let mut p1_best_idx: Option<usize> = None;
        let p1_key = coord_key(segments[edge].start);

        for &link in &p1_neighbors[edge] {
            let angle = compute_angle(&segments[edge], &segments[link], &p1_key);
            angle_pairs.insert((edge, link), angle);
            // Python: max((val, idx) for (idx, val) in enumerate(...))
            // picks maximum angle; on tie, picks higher index
            if angle > p1_best_angle || (angle == p1_best_angle && p1_best_idx.map_or(true, |prev| link > prev)) {
                p1_best_angle = angle;
                p1_best_idx = Some(link);
            }
        }

        if let Some(idx) = p1_best_idx {
            best_p1[edge] = Some((idx, p1_best_angle));
        }

        // Best link at p2 (end endpoint)
        let mut p2_best_angle = 0.0_f64;
        let mut p2_best_idx: Option<usize> = None;
        let p2_key = coord_key(segments[edge].end);

        for &link in &p2_neighbors[edge] {
            let angle = *angle_pairs
                .get(&(edge, link))
                .unwrap_or(&compute_angle(&segments[edge], &segments[link], &p2_key));
            // Store if not already
            angle_pairs.entry((edge, link)).or_insert(angle);
            if angle > p2_best_angle || (angle == p2_best_angle && p2_best_idx.map_or(true, |prev| link > prev)) {
                p2_best_angle = angle;
                p2_best_idx = Some(link);
            }
        }

        if let Some(idx) = p2_best_idx {
            best_p2[edge] = Some((idx, p2_best_angle));
        }
    }

    // 4. Cross-check links: confirm reciprocity and angle threshold.
    //    p1_final[i] = confirmed partner at p1, or None ("line_break")
    //    p2_final[i] = confirmed partner at p2, or None ("line_break")
    let mut p1_final: Vec<Option<usize>> = vec![None; n_segs];
    let mut p2_final: Vec<Option<usize>> = vec![None; n_segs];

    for edge in 0..n_segs {
        // Check p1
        if let Some((bp1, _)) = best_p1[edge] {
            // bp1 must also consider `edge` as its best at whichever endpoint
            let reciprocal = best_p1[bp1].map_or(false, |(b, _)| b == edge)
                || best_p2[bp1].map_or(false, |(b, _)| b == edge);
            let angle = angle_pairs.get(&(edge, bp1)).copied().unwrap_or(0.0);
            if reciprocal && angle > angle_threshold {
                p1_final[edge] = Some(bp1);
            }
        }

        // Check p2
        if let Some((bp2, _)) = best_p2[edge] {
            let reciprocal = best_p1[bp2].map_or(false, |(b, _)| b == edge)
                || best_p2[bp2].map_or(false, |(b, _)| b == edge);
            let angle = angle_pairs.get(&(edge, bp2)).copied().unwrap_or(0.0);
            if reciprocal && angle > angle_threshold {
                p2_final[edge] = Some(bp2);
            }
        }
    }

    // 5. Merge by chain-walking (matches Python's _merge_lines_loop).
    //    Each segment can connect at most once at each end, forming chains.
    let mut seg_to_group: Vec<Option<usize>> = vec![None; n_segs];
    let mut next_group = 0usize;

    for start in 0..n_segs {
        if seg_to_group[start].is_some() {
            continue;
        }

        // Walk from this segment along p1_final/p2_final links
        let mut group_members = vec![start];
        let mut visited = vec![false; n_segs];
        visited[start] = true;

        // Walk forward (following p1_final then p2_final)
        let mut current = start;
        loop {
            let next = if let Some(p) = p1_final[current] {
                if !visited[p] { Some(p) } else { None }
            } else {
                None
            }
            .or_else(|| {
                if let Some(p) = p2_final[current] {
                    if !visited[p] { Some(p) } else { None }
                } else {
                    None
                }
            });

            match next {
                Some(n) => {
                    visited[n] = true;
                    group_members.push(n);
                    current = n;
                }
                None => break,
            }
        }

        // Walk backward from start (following p2_final then p1_final)
        current = start;
        loop {
            let next = if let Some(p) = p2_final[current] {
                if !visited[p] { Some(p) } else { None }
            } else {
                None
            }
            .or_else(|| {
                if let Some(p) = p1_final[current] {
                    if !visited[p] { Some(p) } else { None }
                } else {
                    None
                }
            });

            match next {
                Some(n) => {
                    visited[n] = true;
                    group_members.push(n);
                    current = n;
                }
                None => break,
            }
        }

        let gid = next_group;
        next_group += 1;
        for &m in &group_members {
            seg_to_group[m] = Some(gid);
        }
    }

    // 6. Map segment groups back to original edge indices.
    //    Use the LAST segment's group for each edge (matches Python's dict overwrite).
    let mut edge_to_seg_groups: Vec<Vec<usize>> = vec![vec![]; n];
    for (seg_idx, seg) in segments.iter().enumerate() {
        let gid = seg_to_group[seg_idx].unwrap_or(0);
        edge_to_seg_groups[seg.edge_idx].push(gid);
    }

    // Collect all (edge_idx, group) pairs with last-wins semantics
    // matching Python's inv_edges dict comprehension
    let mut group_to_edges: HashMap<usize, Vec<usize>> = HashMap::new();
    for (seg_idx, seg) in segments.iter().enumerate() {
        let gid = seg_to_group[seg_idx].unwrap_or(0);
        group_to_edges.entry(gid).or_default();
    }

    // Build merged groups: Python deduplicates by sorted segment lists
    // and collects original edge indices per merged group
    let mut merged_groups: Vec<Vec<usize>> = Vec::new(); // group → set of edge indices
    let mut seen_groups: HashMap<Vec<usize>, usize> = HashMap::new();

    for start_seg in 0..n_segs {
        // Walk from this segment to build its group (same as above but we cache)
        let gid = seg_to_group[start_seg].unwrap_or(0);
        // Collect all segments in this group
        if !seen_groups.contains_key(&vec![gid]) {
            // Find all segments with this group id
            let seg_members: Vec<usize> = (0..n_segs)
                .filter(|&s| seg_to_group[s] == Some(gid))
                .collect();
            let edge_members: Vec<usize> = seg_members
                .iter()
                .map(|&s| segments[s].edge_idx)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            seen_groups.insert(vec![gid], merged_groups.len());
            merged_groups.push(edge_members);
        }
    }

    // Assign edge → canonical group via last-wins (matching Python inv_edges)
    let mut edge_groups = vec![0usize; n];
    for (group_idx, edge_members) in merged_groups.iter().enumerate() {
        for &eidx in edge_members {
            edge_groups[eidx] = group_idx;
        }
    }

    // 7. Compute per-group aggregates.
    let mut group_lengths: HashMap<usize, f64> = HashMap::new();
    let mut group_counts: HashMap<usize, usize> = HashMap::new();
    for (edge_idx, &group) in edge_groups.iter().enumerate() {
        let len = geometry_length(geometries, edge_idx);
        *group_lengths.entry(group).or_default() += len;
        *group_counts.entry(group).or_default() += 1;
    }

    // 8. Determine stroke-ends.
    let is_end = compute_stroke_ends(
        &edge_groups, geometries, &p1_final, &p2_final, &segments,
    );

    let stroke_length: Vec<f64> = edge_groups
        .iter()
        .map(|g| *group_lengths.get(g).unwrap_or(&0.0))
        .collect();
    let stroke_count: Vec<usize> = edge_groups
        .iter()
        .map(|g| *group_counts.get(g).unwrap_or(&0))
        .collect();

    let n_p1_confirmed = p1_final.iter().filter(|x| x.is_some()).count();
    let n_p2_confirmed = p2_final.iter().filter(|x| x.is_some()).count();

    CoinsResult {
        group: edge_groups,
        is_end,
        stroke_length,
        stroke_count,
        n_segments: n_segs,
        n_p1_confirmed,
        n_p2_confirmed,
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

/// Coordinate key for hashing using exact float bit equality.
/// This matches Python's exact string comparison for coordinate matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CoordKey {
    x: u64,
    y: u64,
}

fn coord_key(c: [f64; 2]) -> CoordKey {
    CoordKey {
        x: c[0].to_bits(),
        y: c[1].to_bits(),
    }
}

/// Extract two-point segments from all edges.
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

/// Compute interior angle between two segments sharing a node.
///
/// Returns degrees in [0, 180]:
/// - 180 for collinear same-direction continuation
/// - 90 for perpendicular
/// - 0 for opposite directions or no shared node
///
/// Matches Python momepy `_angle_between_two_lines()`.
fn compute_angle(s1: &Segment, s2: &Segment, node: &CoordKey) -> f64 {
    // Get the 4 points
    let a = s1.start;
    let b = s1.end;
    let c = s2.start;
    let d = s2.end;

    // Find the shared point (origin) and two other points
    let points = [a, b, c, d];
    let keys = [coord_key(a), coord_key(b), coord_key(c), coord_key(d)];

    // Origin is the point that appears twice (matching `node`)
    let origin_idx = keys.iter().position(|k| k == node);
    let origin_idx = match origin_idx {
        Some(i) => i,
        None => return 0.0,
    };
    let origin = points[origin_idx];

    // The other point from the same segment as origin
    let partner_of_origin = if origin_idx < 2 {
        // origin is in s1 (a or b), so the "other" from s1 is the non-origin
        if origin_idx == 0 { b } else { a }
    } else {
        // origin is in s2 (c or d), so the "other" from s2 is the non-origin
        if origin_idx == 2 { d } else { c }
    };

    // The other unique point (from the other segment)
    // Find which point in the OTHER segment matches the node
    let other_point = if origin_idx < 2 {
        // origin is in s1, find point in s2 that isn't the shared node
        if coord_key(c) == *node { d } else { c }
    } else {
        // origin is in s2, find point in s1 that isn't the shared node
        if coord_key(a) == *node { b } else { a }
    };

    // Vectors from origin to the two non-shared points
    let v1 = [partner_of_origin[0] - origin[0], partner_of_origin[1] - origin[1]];
    let v2 = [other_point[0] - origin[0], other_point[1] - origin[1]];

    let dot = v1[0] * v2[0] + v1[1] * v2[1];
    let mag1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let mag2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();

    if mag1 < 1e-12 || mag2 < 1e-12 {
        return 0.0;
    }

    // Round to 6 decimal places like Python to match precision
    let cos_theta = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    let cos_rounded = (cos_theta * 1e6).round() / 1e6;
    cos_rounded.acos().to_degrees()
}

/// Get the length of a geometry.
fn geometry_length(geometries: &[GGeometry], idx: usize) -> f64 {
    geometries[idx].length().unwrap_or(0.0)
}

/// Determine which edges are stroke-ends.
/// An edge is a stroke-end if any of its segments has a "line_break" at p1 or p2.
fn compute_stroke_ends(
    edge_groups: &[usize],
    _geometries: &[GGeometry],
    p1_final: &[Option<usize>],
    p2_final: &[Option<usize>],
    segments: &[Segment],
) -> Vec<bool> {
    let n = edge_groups.len();
    let mut is_end = vec![false; n];

    // An edge is a stroke-end if any of its segments has a line_break
    // (None in p1_final or p2_final)
    for (seg_idx, seg) in segments.iter().enumerate() {
        if p1_final[seg_idx].is_none() || p2_final[seg_idx].is_none() {
            is_end[seg.edge_idx] = true;
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
    fn test_coins_perpendicular_merges_low_threshold() {
        // Perpendicular segments: interior angle = 90°.
        // With threshold=30°, 90° > 30° → they SHOULD merge.
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let result = coins(&[g1, g2], 30.0);
        assert_eq!(result.group[0], result.group[1]);
    }

    #[test]
    fn test_coins_perpendicular_splits_high_threshold() {
        // Perpendicular segments: interior angle = 90°.
        // With threshold=120°, 90° < 120° → they should NOT merge.
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let result = coins(&[g1, g2], 120.0);
        assert_ne!(result.group[0], result.group[1]);
    }

    #[test]
    fn test_coins_t_junction() {
        // T-junction: A--B--C with D going up from B
        // A-B-C should form one stroke, D should be separate
        let a_b = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let b_c = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let b_d = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let result = coins(&[a_b, b_c, b_d], 120.0);
        // A-B and B-C are collinear (180°) → should merge
        assert_eq!(result.group[0], result.group[1]);
        // B-D is perpendicular (90° < 120°) → should NOT merge with A-B-C
        assert_ne!(result.group[0], result.group[2]);
    }

    #[test]
    fn test_coins_chain_no_transitive() {
        // A-B, B-C, C-D where B-C angle is below threshold
        // A-B should NOT merge with C-D even though they're connected through B-C
        let a_b = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let b_c = make_line(&[[1.0, 0.0], [1.0, 1.0]]); // 90° turn
        let c_d = make_line(&[[1.0, 1.0], [2.0, 1.0]]);
        let result = coins(&[a_b, b_c, c_d], 120.0);
        // A-B to B-C is 90° < 120° → don't merge
        // B-C to C-D is 90° < 120° → don't merge
        assert_ne!(result.group[0], result.group[1]);
        assert_ne!(result.group[1], result.group[2]);
    }
}
