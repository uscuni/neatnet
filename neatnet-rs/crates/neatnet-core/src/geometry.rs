//! Geometric operations: Voronoi skeleton, angle computation, snap-to-targets.
//!
//! Ports Python `neatnet.geometry`.

use std::collections::HashMap;

use geos::{Geom, Geometry as GGeometry};
use voronoice::{BoundingBox, VoronoiBuilder};

/// Check if a line is within a polygon with a set relative tolerance.
///
/// Returns `true` if the line is entirely within the polygon, or if
/// the intersection length is within `rtol` of the line length.
pub fn is_within(line: &GGeometry, poly: &GGeometry, rtol: f64) -> bool {
    if line.within(poly).unwrap_or(false) {
        return true;
    }
    let intersection = match line.intersection(poly) {
        Ok(i) => i,
        Err(_) => return false,
    };
    let int_len = intersection.length().unwrap_or(0.0);
    let line_len = line.length().unwrap_or(0.0);
    (int_len - line_len).abs() <= rtol
}

/// Compute the angle (in degrees) between two 2-point LineStrings
/// that share a vertex.
///
/// Returns 0.0 if lines are identical or don't share a vertex.
pub fn angle_between_two_lines(line1: &GGeometry, line2: &GGeometry) -> f64 {
    let coords1 = match extract_2pt_coords(line1) {
        Some(c) => c,
        None => return 0.0,
    };
    let coords2 = match extract_2pt_coords(line2) {
        Some(c) => c,
        None => return 0.0,
    };

    // Check if lines share a vertex
    let all_points = [coords1.0, coords1.1, coords2.0, coords2.1];
    let mut counts: HashMap<CoordKey, usize> = HashMap::new();
    for p in &all_points {
        *counts.entry(round_key(*p)).or_default() += 1;
    }

    // Find the shared vertex (count == 2) and the two other points
    let origin = match counts.iter().find(|(_, v)| **v >= 2) {
        Some((k, _)) => *k,
        None => return 0.0, // No shared vertex
    };

    let others: Vec<[f64; 2]> = all_points
        .iter()
        .filter(|p| round_key(**p) != origin)
        .copied()
        .collect();

    if others.len() < 2 {
        return 0.0;
    }

    let ox = origin.x as f64 / 1e8;
    let oy = origin.y as f64 / 1e8;

    let v1 = [others[0][0] - ox, others[0][1] - oy];
    let v2 = [others[1][0] - ox, others[1][1] - oy];

    let dot = v1[0] * v2[0] + v1[1] * v2[1];
    let mag1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
    let mag2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();

    if mag1 < 1e-12 || mag2 < 1e-12 {
        return 0.0;
    }

    let cos_theta = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    cos_theta.acos().to_degrees()
}

/// Generate a Voronoi skeleton from line geometries within a polygon.
///
/// This is the core geometric operation for replacing dual carriageways
/// with centerlines. It:
/// 1. Segmentizes input lines (adds points at max_segment_length intervals)
/// 2. Builds a Voronoi diagram from the points
/// 3. Filters ridges between different input lines
/// 4. Clips results to the polygon
/// 5. Snaps skeleton endpoints to targets
///
/// Returns (edgelines, splitter_points).
pub fn voronoi_skeleton(
    lines: &[GGeometry],
    poly: Option<&GGeometry>,
    snap_to: Option<&[GGeometry]>,
    max_segment_length: f64,
    buffer: Option<f64>,
    secondary_snap_to: Option<&[GGeometry]>,
    clip_limit: f64,
    consolidation_tolerance: Option<f64>,
) -> (Vec<GGeometry>, Vec<GGeometry>) {
    let buffer_dist = buffer.unwrap_or(max_segment_length * 20.0);

    // Get bounding polygon (or compute from lines)
    let working_poly = match poly {
        Some(p) => Clone::clone(p),
        None => {
            // Compute bounding box from all lines
            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            for line in lines {
                if let Ok(env) = line.envelope() {
                    if let Ok(cs) = env.get_coord_seq() {
                        for i in 0..cs.size().unwrap_or(0) {
                            if let (Ok(x), Ok(y)) = (cs.get_x(i), cs.get_y(i)) {
                                min_x = min_x.min(x);
                                min_y = min_y.min(y);
                                max_x = max_x.max(x);
                                max_y = max_y.max(y);
                            }
                        }
                    }
                }
            }
            let wkt = format!(
                "POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))",
                min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y, min_x, min_y
            );
            GGeometry::new_from_wkt(&wkt).unwrap()
        }
    };

    // 1. Segmentize all lines + buffer boundary
    let buffer_geom = working_poly.buffer(buffer_dist, 8).unwrap();
    let buffer_boundary = buffer_geom.boundary().unwrap();

    let all_lines: Vec<&GGeometry> = lines.iter().collect();
    // We'll handle the buffer boundary separately via its segmentized points

    // 2. Extract points from segmentized lines with line-ID tracking
    let mut points: Vec<[f64; 2]> = Vec::new();
    let mut point_line_ids: Vec<usize> = Vec::new();

    for (line_idx, line) in lines.iter().enumerate() {
        let segmentized = match segmentize(line, max_segment_length) {
            Some(s) => s,
            None => continue,
        };
        if let Ok(cs) = segmentized.get_coord_seq() {
            for i in 0..cs.size().unwrap_or(0) {
                if let (Ok(x), Ok(y)) = (cs.get_x(i), cs.get_y(i)) {
                    points.push([x, y]);
                    point_line_ids.push(line_idx);
                }
            }
        }
    }

    // Add buffer boundary points with a special line ID
    let buffer_line_id = lines.len();
    let seg_boundary = segmentize(&buffer_boundary, max_segment_length);
    if let Some(ref seg_b) = seg_boundary {
        if let Ok(cs) = seg_b.get_coord_seq() {
            for i in 0..cs.size().unwrap_or(0) {
                if let (Ok(x), Ok(y)) = (cs.get_x(i), cs.get_y(i)) {
                    points.push([x, y]);
                    point_line_ids.push(buffer_line_id);
                }
            }
        }
    }

    // 3. Remove duplicate points
    let (unique_points, unique_ids) = deduplicate_points(&points, &point_line_ids);

    if unique_points.len() < 3 {
        return (vec![], vec![]);
    }

    // 4. Build Voronoi diagram using voronoice
    let voronoi_points: Vec<voronoice::Point> = unique_points
        .iter()
        .map(|p| voronoice::Point { x: p[0], y: p[1] })
        .collect();

    // Compute bounding box for voronoice
    let mut vmin_x = f64::INFINITY;
    let mut vmin_y = f64::INFINITY;
    let mut vmax_x = f64::NEG_INFINITY;
    let mut vmax_y = f64::NEG_INFINITY;
    for p in &voronoi_points {
        vmin_x = vmin_x.min(p.x);
        vmin_y = vmin_y.min(p.y);
        vmax_x = vmax_x.max(p.x);
        vmax_y = vmax_y.max(p.y);
    }
    let margin = buffer_dist * 2.0;
    let bbox = BoundingBox::new(
        voronoice::Point {
            x: (vmin_x + vmax_x) / 2.0,
            y: (vmin_y + vmax_y) / 2.0,
        },
        (vmax_x - vmin_x) + margin,
        (vmax_y - vmin_y) + margin,
    );

    let voronoi = match VoronoiBuilder::default()
        .set_sites(voronoi_points)
        .set_bounding_box(bbox)
        .build()
    {
        Some(v) => v,
        None => return (vec![], vec![]),
    };

    // 5. Extract Voronoi edges between different input lines
    //    (filtering out same-line ridges and buffer ridges)
    // This is a simplified version - full implementation needs the ridge extraction
    // from the voronoice Delaunay triangulation
    let edgelines = extract_voronoi_edges(
        &voronoi,
        &unique_points,
        &unique_ids,
        &working_poly,
        clip_limit,
        buffer_line_id,
    );

    // 6. Return results
    let splitters = Vec::new(); // TODO: snap_to_targets integration

    (edgelines, splitters)
}

/// Segmentize a geometry: add vertices so no segment exceeds max_length.
pub fn segmentize(geom: &GGeometry, max_length: f64) -> Option<GGeometry> {
    let Ok(cs) = geom.get_coord_seq() else {
        return None;
    };
    let Ok(n) = cs.size() else { return None };
    if n < 2 {
        return Some(Clone::clone(geom));
    }

    let mut new_coords: Vec<[f64; 2]> = Vec::new();

    for i in 0..n - 1 {
        let (Ok(x0), Ok(y0)) = (cs.get_x(i), cs.get_y(i)) else {
            continue;
        };
        let (Ok(x1), Ok(y1)) = (cs.get_x(i + 1), cs.get_y(i + 1)) else {
            continue;
        };

        new_coords.push([x0, y0]);

        let dx = x1 - x0;
        let dy = y1 - y0;
        let seg_len = (dx * dx + dy * dy).sqrt();

        if seg_len > max_length {
            let n_segments = (seg_len / max_length).ceil() as usize;
            for j in 1..n_segments {
                let t = j as f64 / n_segments as f64;
                new_coords.push([x0 + dx * t, y0 + dy * t]);
            }
        }
    }

    // Add last point
    if let (Ok(x), Ok(y)) = (cs.get_x(n - 1), cs.get_y(n - 1)) {
        new_coords.push([x, y]);
    }

    if new_coords.len() < 2 {
        return None;
    }

    let wkt = format!(
        "LINESTRING ({})",
        new_coords
            .iter()
            .map(|c| format!("{} {}", c[0], c[1]))
            .collect::<Vec<_>>()
            .join(", ")
    );
    GGeometry::new_from_wkt(&wkt).ok()
}

/// Snap skeleton edgelines to target geometries.
///
/// For each connected component of edgelines, if it doesn't intersect
/// the polygon boundary, add a shortest line to the nearest snap target.
pub fn snap_to_targets(
    edgelines: &[GGeometry],
    poly: &GGeometry,
    snap_to: &[GGeometry],
    secondary_snap_to: Option<&[GGeometry]>,
) -> (Vec<GGeometry>, Vec<GGeometry>) {
    let mut to_add = Vec::new();
    let mut splitters = Vec::new();

    // For each snap target, find the shortest line from edgelines boundary
    let edgelines_union = union_all(edgelines);
    if let Some(ref union_geom) = edgelines_union {
        if let Ok(boundary) = union_geom.boundary() {
            for target in snap_to {
                if let Some(sl) = make_shortest_line(&boundary, target) {
                    if is_within(&sl, poly, 1e-4) {
                        if let Ok(pt) = sl.get_end_point() {
                            splitters.push(pt);
                        }
                        to_add.push(sl);
                    } else if let Some(secondary) = secondary_snap_to {
                        for sec_target in secondary {
                            if let Some(sl2) = make_shortest_line(&boundary, sec_target) {
                                if let Ok(pt) = sl2.get_end_point() {
                                    splitters.push(pt);
                                }
                                to_add.push(sl2);
                                break;
                            }
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

fn extract_2pt_coords(geom: &GGeometry) -> Option<([f64; 2], [f64; 2])> {
    let cs = geom.get_coord_seq().ok()?;
    let n = cs.size().ok()?;
    if n < 2 {
        return None;
    }
    let a = [cs.get_x(0).ok()?, cs.get_y(0).ok()?];
    let b = [cs.get_x(n - 1).ok()?, cs.get_y(n - 1).ok()?];
    Some((a, b))
}

fn deduplicate_points(
    points: &[[f64; 2]],
    ids: &[usize],
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut seen: HashMap<(i64, i64), usize> = HashMap::new();
    let mut unique_points = Vec::new();
    let mut unique_ids = Vec::new();

    for (p, &id) in points.iter().zip(ids.iter()) {
        let key = ((p[0] * 1e8).round() as i64, (p[1] * 1e8).round() as i64);
        if !seen.contains_key(&key) {
            seen.insert(key, unique_points.len());
            unique_points.push(*p);
            unique_ids.push(id);
        }
    }

    (unique_points, unique_ids)
}

/// Build a LineString connecting the nearest points of two geometries.
///
/// Uses `nearest_points` (geos 10 API) to find the closest pair of points,
/// then returns a 2-point LineString connecting them.
fn make_shortest_line(from: &GGeometry, to: &GGeometry) -> Option<GGeometry> {
    let cs = from.nearest_points(to).ok()?;
    let x0 = cs.get_x(0).ok()?;
    let y0 = cs.get_y(0).ok()?;
    let x1 = cs.get_x(1).ok()?;
    let y1 = cs.get_y(1).ok()?;
    GGeometry::new_from_wkt(&format!("LINESTRING ({} {}, {} {})", x0, y0, x1, y1)).ok()
}

/// Union all geometries into a single geometry.
fn union_all(geoms: &[GGeometry]) -> Option<GGeometry> {
    if geoms.is_empty() {
        return None;
    }
    let mut result = Clone::clone(&geoms[0]);
    for geom in &geoms[1..] {
        result = result.union(geom).ok()?;
    }
    Some(result)
}

/// Extract Voronoi edges between different input line pairs.
///
/// This filters the Voronoi diagram to keep only edges that represent
/// the "skeleton" between pairs of input lines.
fn extract_voronoi_edges(
    voronoi: &voronoice::Voronoi,
    points: &[[f64; 2]],
    point_ids: &[usize],
    poly: &GGeometry,
    clip_limit: f64,
    buffer_line_id: usize,
) -> Vec<GGeometry> {
    // Extract edges from the Voronoi diagram's cell structure
    let edgelines = Vec::new();

    // The voronoice crate gives us cells; we need to extract shared edges
    // between cells belonging to different input lines.
    // This requires iterating over cell neighbors.
    let cells = voronoi.cells();
    let vertices = voronoi.vertices();

    // Build cell adjacency from shared edges
    // For each pair of adjacent cells, if their source points come from
    // different input lines, the shared edge is a skeleton segment
    for (cell_idx, cell) in cells.iter().enumerate() {
        if cell_idx >= point_ids.len() {
            continue;
        }
        let line_id = point_ids[cell_idx];
        if line_id == buffer_line_id {
            continue;
        }

        // Each cell is a list of vertex indices forming the cell polygon
        // Shared edges are between consecutive vertices that are also
        // consecutive in an adjacent cell
        // This is a simplified extraction - for production we'd need
        // the full Delaunay triangulation neighbor info
    }

    // TODO: Complete Voronoi edge extraction with proper ridge filtering
    // This requires access to the Delaunay triangulation which voronoice
    // provides through its internal structure.

    edgelines
}

/// Explode a linestring into constituent pairwise coordinate segments.
///
/// Mirrors Python `artifacts.line_segments()`.
pub fn line_segments(line: &GGeometry) -> Vec<GGeometry> {
    let mut segments = Vec::new();
    let Ok(cs) = line.get_coord_seq() else {
        return segments;
    };
    let Ok(n) = cs.size() else {
        return segments;
    };

    for i in 0..n.saturating_sub(1) {
        let (Ok(x0), Ok(y0)) = (cs.get_x(i), cs.get_y(i)) else {
            continue;
        };
        let (Ok(x1), Ok(y1)) = (cs.get_x(i + 1), cs.get_y(i + 1)) else {
            continue;
        };
        let wkt = format!("LINESTRING ({} {}, {} {})", x0, y0, x1, y1);
        if let Ok(seg) = GGeometry::new_from_wkt(&wkt) {
            segments.push(seg);
        }
    }
    segments
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
    fn test_segmentize() {
        let line = make_line(&[[0.0, 0.0], [10.0, 0.0]]);
        let seg = segmentize(&line, 3.0).unwrap();
        let cs = seg.get_coord_seq().unwrap();
        // Original segment is 10 units → ceil(10/3) = 4 segments → 5 points
        assert!(cs.size().unwrap() >= 4);
    }

    #[test]
    fn test_angle_between_two_lines() {
        let l1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let l2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let angle = angle_between_two_lines(&l1, &l2);
        // Collinear: 180 degrees
        assert!((angle - 180.0).abs() < 1.0);

        let l3 = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let angle2 = angle_between_two_lines(&l1, &l3);
        // Perpendicular: 90 degrees
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
        let poly = GGeometry::new_from_wkt(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
        ).unwrap();
        let inside = make_line(&[[1.0, 1.0], [5.0, 5.0]]);
        assert!(is_within(&inside, &poly, 1e-4));

        let outside = make_line(&[[1.0, 1.0], [15.0, 5.0]]);
        assert!(!is_within(&outside, &poly, 1e-4));
    }
}
