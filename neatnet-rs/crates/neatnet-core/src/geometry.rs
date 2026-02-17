//! Geometric operations: Voronoi skeleton, angle computation, snap-to-targets.
//!
//! Ports Python `neatnet.geometry`.

use std::collections::{BTreeMap, HashMap, HashSet};

use geo::{BooleanOps, Buffer, Centroid, Distance, Euclidean, Length, Simplify, Within};
use geo_types::{Coord, LineString, MultiLineString, Polygon};

use crate::ops;

/// Check if a line is within a polygon with a set relative tolerance.
pub fn is_within(line: &LineString<f64>, poly: &Polygon<f64>, rtol: f64) -> bool {
    if line.is_within(poly) {
        return true;
    }
    let mls = MultiLineString::new(vec![line.clone()]);
    let clipped = poly.clip(&mls, false);
    let int_len: f64 = clipped.0.iter().map(|l| Euclidean.length(l)).sum();
    let line_len = Euclidean.length(line);
    (int_len - line_len).abs() <= rtol
}

/// Compute the angle (in degrees) between two 2-point LineStrings
/// that share a vertex.
pub fn angle_between_two_lines(line1: &LineString<f64>, line2: &LineString<f64>) -> f64 {
    let coords1 = match extract_2pt_coords(line1) {
        Some(c) => c,
        None => return 0.0,
    };
    let coords2 = match extract_2pt_coords(line2) {
        Some(c) => c,
        None => return 0.0,
    };

    let all_points = [coords1.0, coords1.1, coords2.0, coords2.1];
    let mut counts: HashMap<CoordKey, usize> = HashMap::new();
    for p in &all_points {
        *counts.entry(round_key(*p)).or_default() += 1;
    }

    let origin = match counts.iter().find(|(_, v)| **v >= 2) {
        Some((k, _)) => *k,
        None => return 0.0,
    };

    let others: Vec<[f64; 2]> = all_points
        .iter()
        .filter(|p| round_key(**p) != origin)
        .copied()
        .collect();

    if others.len() < 2 { return 0.0; }

    let ox = origin.x as f64 / 1e8;
    let oy = origin.y as f64 / 1e8;

    let v1 = [others[0][0] - ox, others[0][1] - oy];
    let v2 = [others[1][0] - ox, others[1][1] - oy];

    let dot = v1[0] * v2[0] + v1[1] * v2[1];
    let mag1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let mag2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();

    if mag1 < 1e-12 || mag2 < 1e-12 { return 0.0; }

    let cos_theta = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

/// Generate a Voronoi skeleton from line geometries within a polygon.
pub fn voronoi_skeleton(
    lines: &[LineString<f64>],
    poly: Option<&Polygon<f64>>,
    snap_to: Option<&[LineString<f64>]>,
    max_segment_length: f64,
    buffer: Option<f64>,
    secondary_snap_to: Option<&[LineString<f64>]>,
    clip_limit: f64,
    _consolidation_tolerance: Option<f64>,
) -> (Vec<LineString<f64>>, Vec<LineString<f64>>) {
    let buffer_dist = buffer.unwrap_or(max_segment_length * 20.0);

    // Get bounding polygon
    let working_poly = match poly {
        Some(p) => p.clone(),
        None => {
            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for line in lines {
                for c in &line.0 {
                    min_x = min_x.min(c.x);
                    min_y = min_y.min(c.y);
                    max_x = max_x.max(c.x);
                    max_y = max_y.max(c.y);
                }
            }
            Polygon::new(
                LineString::new(vec![
                    Coord { x: min_x, y: min_y },
                    Coord { x: max_x, y: min_y },
                    Coord { x: max_x, y: max_y },
                    Coord { x: min_x, y: max_y },
                    Coord { x: min_x, y: min_y },
                ]),
                vec![],
            )
        }
    };

    // 1. Buffer boundary
    let buffer_mp = working_poly.buffer(buffer_dist);
    let buffer_geom = match buffer_mp.0.into_iter().next() {
        Some(b) => b,
        None => return (vec![], vec![]),
    };
    let buffer_boundary = buffer_geom.exterior().clone();

    // 2. Extract points from segmentized lines
    let mut points: Vec<[f64; 2]> = Vec::new();
    let mut point_line_ids: Vec<usize> = Vec::new();

    for (line_idx, line) in lines.iter().enumerate() {
        let segmentized = segmentize(line, max_segment_length);
        for c in &segmentized.0 {
            points.push([c.x, c.y]);
            point_line_ids.push(line_idx);
        }
    }

    // Add buffer boundary points (use coarser segmentation for performance —
    // the buffer boundary only needs to define the Voronoi cell boundary,
    // not fine details)
    let buffer_line_id = lines.len();
    let buffer_seg_length = max_segment_length * 5.0;
    let seg_boundary = segmentize(&buffer_boundary, buffer_seg_length);
    for c in &seg_boundary.0 {
        points.push([c.x, c.y]);
        point_line_ids.push(buffer_line_id);
    }

    // 3. Remove duplicate points
    let (unique_points, unique_ids) = deduplicate_points(&points, &point_line_ids);
    if unique_points.len() < 3 {
        return (vec![], vec![]);
    }

    // 4. Build Delaunay triangulation
    let del_points: Vec<delaunator::Point> = unique_points
        .iter()
        .map(|p| delaunator::Point { x: p[0], y: p[1] })
        .collect();
    let triangulation = delaunator::triangulate(&del_points);

    // 5. Extract ridges between different input lines
    let ridges = extract_ridges(&triangulation, &del_points, &unique_ids, buffer_line_id);
    if ridges.is_empty() {
        return (vec![], vec![]);
    }

    // 6. Compute clip polygon
    let mic_radius = approximate_mic_radius(&working_poly);
    let dist = clip_limit.min(mic_radius * 0.4);
    let limit_mp = working_poly.buffer(-dist);
    let limit = match limit_mp.0.into_iter().next() {
        Some(l) if l.exterior().0.len() >= 4 => l,
        _ => working_poly.clone(),
    };

    // 7. Build edgelines
    let mut edgelines = build_edgelines(&ridges, &limit, lines);
    edgelines.retain(|e| e.0.len() >= 2 && Euclidean.length(e) > 0.0);

    if edgelines.is_empty() {
        return (vec![], vec![]);
    }

    // 9. Snapping
    let mut splitters = Vec::new();
    let mut to_add = Vec::new();

    match snap_to {
        None => {
            if let Some(boundary_line) = union_all_boundary(&edgelines) {
                let poly_boundary = working_poly.exterior();
                if let Some(sl) = make_shortest_line_ls(&boundary_line, poly_boundary) {
                    splitters.push(sl.clone());
                    to_add.push(sl);
                }
            }
        }
        Some(targets) => {
            let (additions, splits) =
                snap_to_targets(&edgelines, &working_poly, targets, secondary_snap_to);
            to_add.extend(additions);
            splitters.extend(splits);
        }
    }

    edgelines.extend(to_add);

    // 10. Simplify
    edgelines = edgelines
        .iter()
        .map(|e| e.simplify(max_segment_length))
        .collect();

    // 11. Line merge and explode
    edgelines = ops::line_merge(&edgelines);
    edgelines.retain(|e| e.0.len() >= 2 && Euclidean.length(e) > 0.0);

    (edgelines, splitters)
}

/// Segmentize a LineString: add vertices so no segment exceeds max_length.
pub fn segmentize(geom: &LineString<f64>, max_length: f64) -> LineString<f64> {
    let coords = &geom.0;
    if coords.len() < 2 {
        return geom.clone();
    }

    let mut new_coords: Vec<Coord<f64>> = Vec::new();

    for i in 0..coords.len() - 1 {
        let (x0, y0) = (coords[i].x, coords[i].y);
        let (x1, y1) = (coords[i + 1].x, coords[i + 1].y);
        new_coords.push(Coord { x: x0, y: y0 });

        let dx = x1 - x0;
        let dy = y1 - y0;
        let seg_len = (dx * dx + dy * dy).sqrt();

        if seg_len > max_length {
            let n_segments = (seg_len / max_length).ceil() as usize;
            for j in 1..n_segments {
                let t = j as f64 / n_segments as f64;
                new_coords.push(Coord { x: x0 + dx * t, y: y0 + dy * t });
            }
        }
    }

    new_coords.push(*coords.last().unwrap());
    LineString::new(new_coords)
}

/// Snap skeleton edgelines to target geometries.
pub fn snap_to_targets(
    edgelines: &[LineString<f64>],
    poly: &Polygon<f64>,
    snap_to: &[LineString<f64>],
    secondary_snap_to: Option<&[LineString<f64>]>,
) -> (Vec<LineString<f64>>, Vec<LineString<f64>>) {
    let mut to_add = Vec::new();
    let mut splitters = Vec::new();

    if let Some(boundary_line) = union_all_boundary(edgelines) {
        for target in snap_to {
            if let Some(sl) = make_shortest_line_ls(&boundary_line, target) {
                if is_within(&sl, poly, 1e-4) {
                    splitters.push(sl.clone());
                    to_add.push(sl);
                } else if let Some(secondary) = secondary_snap_to {
                    for sec_target in secondary {
                        if let Some(sl2) = make_shortest_line_ls(&boundary_line, sec_target) {
                            splitters.push(sl2.clone());
                            to_add.push(sl2);
                            break;
                        }
                    }
                }
            }
        }
    }

    (to_add, splitters)
}

// ─── Internal helpers ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CoordKey {
    x: i64,
    y: i64,
}

fn round_key(c: [f64; 2]) -> CoordKey {
    CoordKey {
        x: (c[0] * 1e8).round() as i64,
        y: (c[1] * 1e8).round() as i64,
    }
}

fn extract_2pt_coords(geom: &LineString<f64>) -> Option<([f64; 2], [f64; 2])> {
    let coords = &geom.0;
    if coords.len() < 2 { return None; }
    let a = [coords[0].x, coords[0].y];
    let b = [coords[coords.len() - 1].x, coords[coords.len() - 1].y];
    Some((a, b))
}

fn deduplicate_points(
    points: &[[f64; 2]],
    ids: &[usize],
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut counts: HashMap<(u64, u64), usize> = HashMap::new();
    for p in points {
        let key = (p[0].to_bits(), p[1].to_bits());
        *counts.entry(key).or_default() += 1;
    }

    let mut unique_points = Vec::new();
    let mut unique_ids = Vec::new();
    for (p, &id) in points.iter().zip(ids.iter()) {
        let key = (p[0].to_bits(), p[1].to_bits());
        if counts[&key] == 1 {
            unique_points.push(*p);
            unique_ids.push(id);
        }
    }
    (unique_points, unique_ids)
}

/// Build a LineString connecting the nearest points of two LineStrings.
fn make_shortest_line_ls(from: &LineString<f64>, to: &LineString<f64>) -> Option<LineString<f64>> {
    let (pa, pb) = ops::nearest_points(from, to)?;
    let dist = ((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2)).sqrt();
    if dist < 1e-10 { return None; }
    Some(LineString::new(vec![pa, pb]))
}

/// Get the boundary of a union of linestrings (their endpoints).
fn union_all_boundary(lines: &[LineString<f64>]) -> Option<LineString<f64>> {
    // The "boundary" of a set of linestrings is their endpoint collection.
    // For snapping, we just need a linestring containing all endpoints.
    let mut pts: Vec<Coord<f64>> = Vec::new();
    for line in lines {
        if line.0.len() >= 2 {
            pts.push(line.0[0]);
            pts.push(*line.0.last().unwrap());
        }
    }
    if pts.len() < 2 { return None; }
    // Deduplicate and return as a linestring (for nearest_points)
    let mut seen = HashSet::new();
    pts.retain(|c| {
        let key = ((c.x * 1e8).round() as i64, (c.y * 1e8).round() as i64);
        seen.insert(key)
    });
    if pts.len() < 2 {
        // If only one unique point, duplicate it
        pts.push(pts[0]);
    }
    Some(LineString::new(pts))
}

struct Ridge {
    line_a: usize,
    line_b: usize,
    vertex_a: [f64; 2],
    vertex_b: [f64; 2],
}

fn extract_ridges(
    tri: &delaunator::Triangulation,
    points: &[delaunator::Point],
    point_ids: &[usize],
    buffer_line_id: usize,
) -> Vec<Ridge> {
    let n_halfedges = tri.halfedges.len();
    let mut ridges = Vec::new();
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

    for e in 0..n_halfedges {
        let opp = tri.halfedges[e];
        if opp == delaunator::EMPTY { continue; }
        if e > opp { continue; }

        let p_a = tri.triangles[e];
        let p_b = tri.triangles[next_halfedge(e)];

        let line_a = point_ids[p_a];
        let line_b = point_ids[p_b];
        if line_a == line_b { continue; }
        if line_a == buffer_line_id || line_b == buffer_line_id { continue; }

        let edge_key = if p_a < p_b { (p_a, p_b) } else { (p_b, p_a) };
        if !seen_edges.insert(edge_key) { continue; }

        let tri_1 = e / 3;
        let tri_2 = opp / 3;
        let cc1 = circumcenter(
            &points[tri.triangles[tri_1 * 3]],
            &points[tri.triangles[tri_1 * 3 + 1]],
            &points[tri.triangles[tri_1 * 3 + 2]],
        );
        let cc2 = circumcenter(
            &points[tri.triangles[tri_2 * 3]],
            &points[tri.triangles[tri_2 * 3 + 1]],
            &points[tri.triangles[tri_2 * 3 + 2]],
        );

        ridges.push(Ridge { line_a, line_b, vertex_a: cc1, vertex_b: cc2 });
    }

    ridges
}

fn next_halfedge(e: usize) -> usize {
    if e % 3 == 2 { e - 2 } else { e + 1 }
}

fn circumcenter(a: &delaunator::Point, b: &delaunator::Point, c: &delaunator::Point) -> [f64; 2] {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < 1e-20 {
        return [(a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0];
    }
    let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
        + (b.x * b.x + b.y * b.y) * (c.y - a.y)
        + (c.x * c.x + c.y * c.y) * (a.y - b.y)) / d;
    let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
        + (b.x * b.x + b.y * b.y) * (a.x - c.x)
        + (c.x * c.x + c.y * c.y) * (b.x - a.x)) / d;
    [ux, uy]
}

fn approximate_mic_radius(poly: &Polygon<f64>) -> f64 {
    let centroid = match poly.centroid() {
        Some(c) => c,
        None => return 1.0,
    };
    let boundary = poly.exterior();
    Euclidean.distance(&centroid, boundary)
}

fn build_edgelines(
    ridges: &[Ridge],
    limit: &Polygon<f64>,
    lines: &[LineString<f64>],
) -> Vec<LineString<f64>> {
    use rayon::prelude::*;

    let mut ridge_groups: BTreeMap<(usize, usize), Vec<&Ridge>> = BTreeMap::new();
    for ridge in ridges {
        let key = if ridge.line_a < ridge.line_b {
            (ridge.line_a, ridge.line_b)
        } else {
            (ridge.line_b, ridge.line_a)
        };
        ridge_groups.entry(key).or_default().push(ridge);
    }

    // Collect groups into a Vec for parallel processing
    let groups: Vec<((usize, usize), Vec<&Ridge>)> = ridge_groups.into_iter().collect();

    groups.par_iter()
        .filter_map(|((line_a, line_b), group)| {
            let mut segments: Vec<LineString<f64>> = Vec::new();
            for ridge in group {
                let seg = LineString::new(vec![
                    Coord { x: ridge.vertex_a[0], y: ridge.vertex_a[1] },
                    Coord { x: ridge.vertex_b[0], y: ridge.vertex_b[1] },
                ]);
                if Euclidean.length(&seg) > 0.0 {
                    segments.push(seg);
                }
            }

            if segments.is_empty() { return None; }

            // Line merge all ridge segments
            let merged = ops::line_merge(&segments);
            let edgeline = if merged.len() == 1 {
                merged.into_iter().next().unwrap()
            } else if merged.is_empty() {
                return None;
            } else {
                // Keep the longest
                merged.into_iter().max_by(|a, b| {
                    Euclidean.length(a).partial_cmp(&Euclidean.length(b)).unwrap()
                }).unwrap()
            };

            // Clip to limit polygon
            let edgeline = clip_edgeline(&edgeline, limit);
            if edgeline.0.len() < 2 { return None; }

            // Handle shared vertex connections
            if *line_a < lines.len() && *line_b < lines.len() {
                let edgeline = add_shared_vertex_connections(
                    edgeline, &lines[*line_a], &lines[*line_b], lines.len(),
                );
                Some(edgeline)
            } else {
                Some(edgeline)
            }
        })
        .collect()
}

fn clip_edgeline(edgeline: &LineString<f64>, limit: &Polygon<f64>) -> LineString<f64> {
    if edgeline.is_within(limit) {
        return edgeline.clone();
    }

    // Clip line to polygon using BooleanOps::clip
    let mls = MultiLineString::new(vec![edgeline.clone()]);
    let clipped = limit.clip(&mls, false);

    if clipped.0.is_empty() {
        return LineString::new(vec![]);
    }

    // Keep the longest part (remove slivers)
    clipped.0.into_iter()
        .max_by(|a, b| Euclidean.length(a).partial_cmp(&Euclidean.length(b)).unwrap())
        .unwrap_or_else(|| LineString::new(vec![]))
}

fn add_shared_vertex_connections(
    edgeline: LineString<f64>,
    line_a: &LineString<f64>,
    line_b: &LineString<f64>,
    n_lines: usize,
) -> LineString<f64> {
    // Find shared endpoints between the two input lines
    let a_start = line_a.0.first();
    let a_end = line_a.0.last();
    let b_start = line_b.0.first();
    let b_end = line_b.0.last();

    let mut shared_pts: Vec<Coord<f64>> = Vec::new();
    let eps = 1e-6;
    for pa in [a_start, a_end].iter().flatten() {
        for pb in [b_start, b_end].iter().flatten() {
            let dx = pa.x - pb.x;
            let dy = pa.y - pb.y;
            if (dx * dx + dy * dy).sqrt() < eps {
                shared_pts.push(**pa);
            }
        }
    }

    if shared_pts.is_empty() {
        return edgeline;
    }

    if shared_pts.len() == 2 && n_lines != 2 {
        return edgeline;
    }

    // Add shortest lines from shared points to edgeline boundary
    let mut all_lines = vec![edgeline.clone()];
    let boundary_pts = LineString::new(vec![
        edgeline.0[0],
        *edgeline.0.last().unwrap(),
    ]);

    for pt in &shared_pts {
        let pt_line = LineString::new(vec![*pt, *pt]);
        if let Some(sl) = make_shortest_line_ls(&pt_line, &boundary_pts) {
            if Euclidean.length(&sl) > 1e-10 {
                all_lines.push(sl);
            }
        }
    }

    if all_lines.len() == 1 {
        return edgeline;
    }

    // Merge all
    let merged = ops::line_merge(&all_lines);
    if merged.len() == 1 {
        merged.into_iter().next().unwrap()
    } else {
        edgeline
    }
}

/// Explode a linestring into constituent pairwise coordinate segments.
pub fn line_segments(line: &LineString<f64>) -> Vec<LineString<f64>> {
    let coords = &line.0;
    let mut segments = Vec::new();
    for i in 0..coords.len().saturating_sub(1) {
        segments.push(LineString::new(vec![coords[i], coords[i + 1]]));
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_line(coords: &[[f64; 2]]) -> LineString<f64> {
        LineString::new(coords.iter().map(|c| Coord { x: c[0], y: c[1] }).collect())
    }

    #[test]
    fn test_segmentize() {
        let line = make_line(&[[0.0, 0.0], [10.0, 0.0]]);
        let seg = segmentize(&line, 3.0);
        assert!(seg.0.len() >= 4);
    }

    #[test]
    fn test_angle_between_two_lines() {
        let l1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let l2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let angle = angle_between_two_lines(&l1, &l2);
        assert!((angle - 180.0).abs() < 1.0);

        let l3 = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let angle2 = angle_between_two_lines(&l1, &l3);
        assert!((angle2 - 90.0).abs() < 1.0);
    }

    #[test]
    fn test_line_segments() {
        let line = make_line(&[[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]]);
        let segs = line_segments(&line);
        assert_eq!(segs.len(), 2);
    }

    #[test]
    fn test_is_within() {
        let poly = Polygon::new(
            LineString::new(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        );
        let inside = make_line(&[[1.0, 1.0], [5.0, 5.0]]);
        assert!(is_within(&inside, &poly, 1e-4));

        let outside = make_line(&[[1.0, 1.0], [15.0, 5.0]]);
        assert!(!is_within(&outside, &poly, 1e-4));
    }
}
