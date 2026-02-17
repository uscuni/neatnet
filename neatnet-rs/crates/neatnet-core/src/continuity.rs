//! COINS (Continuity in Street Networks) algorithm.
//!
//! Ports the `momepy.COINS` implementation: given a set of LineString
//! geometries and an angle threshold, assigns each edge to a "stroke"
//! group by greedily pairing edges at shared endpoints with the
//! highest interior angle (most collinear continuation), provided
//! the angle exceeds the threshold.

use std::collections::HashMap;

use geo::{Euclidean, Length};
use geo_types::LineString;
use rayon::prelude::*;

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
pub fn coins(geometries: &[LineString<f64>], angle_threshold: f64) -> CoinsResult {
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

    // 2. Build endpoint -> segment adjacency.
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

    // Build neighbor lists — read-only lookups into point_to_segs, parallelized.
    let (p1_neighbors, p2_neighbors): (Vec<Vec<usize>>, Vec<Vec<usize>>) = (0..n_segs)
        .into_par_iter()
        .map(|idx| {
            let seg = &segments[idx];
            let p1n = match point_to_segs.get(&coord_key(seg.start)) {
                Some(others) => others.iter().copied().filter(|j| *j != idx).collect(),
                None => vec![],
            };
            let p2n = match point_to_segs.get(&coord_key(seg.end)) {
                Some(others) => others.iter().copied().filter(|j| *j != idx).collect(),
                None => vec![],
            };
            (p1n, p2n)
        })
        .unzip();

    // 3. Compute best link at each endpoint — fully parallel per edge.
    //    Each edge independently evaluates angles with its neighbors.
    //    No shared mutable state: we compute angles fresh (cheap arithmetic).
    let (best_p1, best_p2): (Vec<Option<(usize, f64)>>, Vec<Option<(usize, f64)>>) = (0..n_segs)
        .into_par_iter()
        .map(|edge| {
            let seg = &segments[edge];

            // Best link at p1 (start endpoint)
            let p1_key = coord_key(seg.start);
            let bp1 = p1_neighbors[edge]
                .iter()
                .map(|link| {
                    let angle = compute_angle(seg, &segments[*link], &p1_key);
                    (*link, angle)
                })
                .fold(None::<(usize, f64)>, |best, (link, angle)| match best {
                    None => Some((link, angle)),
                    Some((prev_link, prev_angle)) => {
                        if angle > prev_angle
                            || (angle == prev_angle && link > prev_link)
                        {
                            Some((link, angle))
                        } else {
                            Some((prev_link, prev_angle))
                        }
                    }
                });

            // Best link at p2 (end endpoint)
            let p2_key = coord_key(seg.end);
            let bp2 = p2_neighbors[edge]
                .iter()
                .map(|link| {
                    let angle = compute_angle(seg, &segments[*link], &p2_key);
                    (*link, angle)
                })
                .fold(None::<(usize, f64)>, |best, (link, angle)| match best {
                    None => Some((link, angle)),
                    Some((prev_link, prev_angle)) => {
                        if angle > prev_angle
                            || (angle == prev_angle && link > prev_link)
                        {
                            Some((link, angle))
                        } else {
                            Some((prev_link, prev_angle))
                        }
                    }
                });

            (bp1, bp2)
        })
        .unzip();

    // 4. Cross-check links: confirm reciprocity and angle threshold — parallel.
    let (p1_final, p2_final): (Vec<Option<usize>>, Vec<Option<usize>>) = (0..n_segs)
        .into_par_iter()
        .map(|edge| {
            let pf1 = match best_p1[edge] {
                Some((bp1, _)) => {
                    let reciprocal = best_p1[bp1].map_or(false, |(b, _)| b == edge)
                        || best_p2[bp1].map_or(false, |(b, _)| b == edge);
                    let p1_key = coord_key(segments[edge].start);
                    let angle = compute_angle(&segments[edge], &segments[bp1], &p1_key);
                    if reciprocal && angle > angle_threshold {
                        Some(bp1)
                    } else {
                        None
                    }
                }
                None => None,
            };
            let pf2 = match best_p2[edge] {
                Some((bp2, _)) => {
                    let reciprocal = best_p1[bp2].map_or(false, |(b, _)| b == edge)
                        || best_p2[bp2].map_or(false, |(b, _)| b == edge);
                    let p2_key = coord_key(segments[edge].end);
                    let angle = compute_angle(&segments[edge], &segments[bp2], &p2_key);
                    if reciprocal && angle > angle_threshold {
                        Some(bp2)
                    } else {
                        None
                    }
                }
                None => None,
            };
            (pf1, pf2)
        })
        .unzip();

    // 5. Merge by chain-walking (sequential — inherently serial graph traversal).
    let mut seg_to_group: Vec<Option<usize>> = vec![None; n_segs];
    let mut next_group = 0usize;

    for start in 0..n_segs {
        if seg_to_group[start].is_some() {
            continue;
        }
        let mut group_members = vec![start];
        let mut visited = vec![false; n_segs];
        visited[start] = true;

        let mut current = start;
        loop {
            let next = p1_final[current]
                .filter(|p| !visited[*p])
                .or_else(|| p2_final[current].filter(|p| !visited[*p]));
            match next {
                Some(n) => {
                    visited[n] = true;
                    group_members.push(n);
                    current = n;
                }
                None => break,
            }
        }

        current = start;
        loop {
            let next = p2_final[current]
                .filter(|p| !visited[*p])
                .or_else(|| p1_final[current].filter(|p| !visited[*p]));
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
    let mut seen_groups: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut merged_groups: Vec<Vec<usize>> = Vec::new();

    for start_seg in 0..n_segs {
        let gid = seg_to_group[start_seg].unwrap_or(0);
        if !seen_groups.contains_key(&vec![gid]) {
            let seg_members: Vec<usize> = (0..n_segs)
                .filter(|s| seg_to_group[*s] == Some(gid))
                .collect();
            let edge_members: Vec<usize> = seg_members
                .iter()
                .map(|s| segments[*s].edge_idx)
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect();
            seen_groups.insert(vec![gid], merged_groups.len());
            merged_groups.push(edge_members);
        }
    }

    let mut edge_groups = vec![0usize; n];
    for (group_idx, edge_members) in merged_groups.iter().enumerate() {
        for &eidx in edge_members {
            edge_groups[eidx] = group_idx;
        }
    }

    // 7. Compute per-group aggregates — parallel length computation.
    let edge_lengths: Vec<f64> = geometries
        .par_iter()
        .map(|g| Euclidean.length(g))
        .collect();

    let mut group_lengths: HashMap<usize, f64> = HashMap::new();
    let mut group_counts: HashMap<usize, usize> = HashMap::new();
    for (edge_idx, &group) in edge_groups.iter().enumerate() {
        *group_lengths.entry(group).or_default() += edge_lengths[edge_idx];
        *group_counts.entry(group).or_default() += 1;
    }

    // 8. Determine stroke-ends.
    let is_end = compute_stroke_ends(
        &edge_groups, &p1_final, &p2_final, &segments,
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

/// CES classification info for a single artifact polygon.
#[derive(Debug, Clone)]
pub struct CesInfo {
    pub stroke_count: usize,
    pub c: usize,
    pub e: usize,
    pub s: usize,
}

/// Classify strokes within artifact polygons using CES typology.
pub fn get_stroke_info(
    artifact_geoms: &[geo_types::Polygon<f64>],
    street_geoms: &[LineString<f64>],
    coins_result: &CoinsResult,
) -> Vec<CesInfo> {
    use geo::Relate;
    let tree = crate::spatial::build_rtree(street_geoms);

    let mut result = Vec::with_capacity(artifact_geoms.len());

    for artifact in artifact_geoms {
        let candidates = envelope_query_indices_poly(&tree, artifact);
        let mut covered_edges: Vec<usize> = Vec::new();
        for idx in candidates {
            // Check if edge is covered by the artifact
            let de9im = artifact.relate(&street_geoms[idx]);
            if de9im.is_covers() {
                covered_edges.push(idx);
            }
        }

        if covered_edges.is_empty() {
            result.push(CesInfo { stroke_count: 0, c: 0, e: 0, s: 0 });
            continue;
        }

        let covered_groups: std::collections::HashSet<usize> =
            covered_edges.iter().map(|&i| coins_result.group[i]).collect();
        let stroke_count = covered_groups.len();

        if covered_groups.len() == 1 {
            let total_in_group = coins_result.stroke_count[covered_edges[0]];
            if covered_edges.len() == total_in_group {
                result.push(CesInfo { stroke_count, c: 0, e: 0, s: 1 });
                continue;
            }
        }

        let end_edges: Vec<usize> = covered_edges
            .iter()
            .filter(|&&i| coins_result.is_end[i])
            .copied()
            .collect();
        let end_groups: std::collections::HashSet<usize> =
            end_edges.iter().map(|&i| coins_result.group[i]).collect();

        let c_count = covered_groups.iter().filter(|g| !end_groups.contains(g)).count();

        let mut s_count = 0;
        let mut e_count = 0;
        let mut visited_groups = std::collections::HashSet::new();

        for &edge_idx in &end_edges {
            let group = coins_result.group[edge_idx];
            if visited_groups.contains(&group) {
                continue;
            }
            let total_in_group = coins_result.stroke_count[edge_idx];
            let count_in_artifact = covered_edges
                .iter()
                .filter(|&&i| coins_result.group[i] == group)
                .count();
            if count_in_artifact == total_in_group {
                s_count += 1;
                visited_groups.insert(group);
            } else {
                e_count += 1;
            }
        }

        result.push(CesInfo { stroke_count, c: c_count, e: e_count, s: s_count });
    }

    result
}

/// Query R-tree for LineString indices near a polygon's bounding box.
fn envelope_query_indices_poly(
    tree: &rstar::RTree<crate::spatial::IndexedEnvelope>,
    poly: &geo_types::Polygon<f64>,
) -> Vec<usize> {
    use geo::BoundingRect;
    match poly.bounding_rect() {
        Some(rect) => crate::spatial::query_envelope(
            tree,
            [rect.min().x, rect.min().y],
            [rect.max().x, rect.max().y],
        ),
        None => vec![],
    }
}

// ─── Internal helpers ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Segment {
    start: [f64; 2],
    end: [f64; 2],
    edge_idx: usize,
}

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

fn extract_segments(geometries: &[LineString<f64>]) -> Vec<Segment> {
    let mut segments = Vec::new();
    for (edge_idx, geom) in geometries.iter().enumerate() {
        let coords = &geom.0;
        if coords.len() < 2 {
            continue;
        }
        for i in 0..coords.len() - 1 {
            segments.push(Segment {
                start: [coords[i].x, coords[i].y],
                end: [coords[i + 1].x, coords[i + 1].y],
                edge_idx,
            });
        }
    }
    segments
}

fn compute_angle(s1: &Segment, s2: &Segment, node: &CoordKey) -> f64 {
    let a = s1.start;
    let b = s1.end;
    let c = s2.start;
    let d = s2.end;

    let points = [a, b, c, d];
    let keys = [coord_key(a), coord_key(b), coord_key(c), coord_key(d)];

    let origin_idx = match keys.iter().position(|k| k == node) {
        Some(i) => i,
        None => return 0.0,
    };
    let origin = points[origin_idx];

    let partner_of_origin = if origin_idx < 2 {
        if origin_idx == 0 { b } else { a }
    } else {
        if origin_idx == 2 { d } else { c }
    };

    let other_point = if origin_idx < 2 {
        if coord_key(c) == *node { d } else { c }
    } else {
        if coord_key(a) == *node { b } else { a }
    };

    let v1 = [partner_of_origin[0] - origin[0], partner_of_origin[1] - origin[1]];
    let v2 = [other_point[0] - origin[0], other_point[1] - origin[1]];

    let dot = v1[0] * v2[0] + v1[1] * v2[1];
    let mag1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let mag2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();

    if mag1 < 1e-12 || mag2 < 1e-12 {
        return 0.0;
    }

    let cos_theta = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    let cos_rounded = (cos_theta * 1e6).round() / 1e6;
    cos_rounded.acos().to_degrees()
}

fn compute_stroke_ends(
    edge_groups: &[usize],
    p1_final: &[Option<usize>],
    p2_final: &[Option<usize>],
    segments: &[Segment],
) -> Vec<bool> {
    let n = edge_groups.len();
    let mut is_end = vec![false; n];
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
    use geo_types::Coord;

    fn make_line(coords: &[[f64; 2]]) -> LineString<f64> {
        LineString::new(coords.iter().map(|c| Coord { x: c[0], y: c[1] }).collect())
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
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let result = coins(&[g1, g2], 120.0);
        assert_eq!(result.group[0], result.group[1]);
        assert_eq!(result.stroke_count[0], 2);
    }

    #[test]
    fn test_coins_perpendicular_merges_low_threshold() {
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let result = coins(&[g1, g2], 30.0);
        assert_eq!(result.group[0], result.group[1]);
    }

    #[test]
    fn test_coins_perpendicular_splits_high_threshold() {
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let result = coins(&[g1, g2], 120.0);
        assert_ne!(result.group[0], result.group[1]);
    }

    #[test]
    fn test_coins_t_junction() {
        let a_b = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let b_c = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let b_d = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let result = coins(&[a_b, b_c, b_d], 120.0);
        assert_eq!(result.group[0], result.group[1]);
        assert_ne!(result.group[0], result.group[2]);
    }

    #[test]
    fn test_coins_chain_no_transitive() {
        let a_b = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let b_c = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let c_d = make_line(&[[1.0, 1.0], [2.0, 1.0]]);
        let result = coins(&[a_b, b_c, c_d], 120.0);
        assert_ne!(result.group[0], result.group[1]);
        assert_ne!(result.group[1], result.group[2]);
    }
}
