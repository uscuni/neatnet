//! Geometric operations: Voronoi skeleton, angle computation, snap-to-targets.
//!
//! Ports Python `neatnet.geometry`.

use std::collections::{BTreeMap, HashMap, HashSet};

use geos::{Geom, Geometry as GGeometry, GeometryTypes};

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
/// 2. Builds a Delaunay triangulation from the points
/// 3. Extracts Voronoi ridges (dual edges) between different input lines
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
    _consolidation_tolerance: Option<f64>,
) -> (Vec<GGeometry>, Vec<GGeometry>) {
    let buffer_dist = buffer.unwrap_or(max_segment_length * 20.0);

    // Get bounding polygon (or compute from lines)
    let working_poly = match poly {
        Some(p) => Clone::clone(p),
        None => {
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
    let buffer_geom = match working_poly.buffer(buffer_dist, 8) {
        Ok(b) => b,
        Err(_) => return (vec![], vec![]),
    };
    let buffer_boundary = match buffer_geom.boundary() {
        Ok(b) => b,
        Err(_) => return (vec![], vec![]),
    };

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

    // 3. Remove duplicate points (remove ALL copies of any duplicated point,
    //    matching Python's behavior for Voronoi input)
    let (unique_points, unique_ids) = deduplicate_points(&points, &point_line_ids);

    if unique_points.len() < 3 {
        return (vec![], vec![]);
    }

    // 4. Build Delaunay triangulation and extract Voronoi ridges.
    // We use the Delaunay dual because each Delaunay edge maps directly to
    // the two input points it connects, giving exact line-pair assignments
    // (unlike GEOS voronoi which requires approximate nearest-neighbor mapping).
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

    // 6. Compute clip polygon: poly.buffer(-dist) where dist = min(clip_limit, mic_radius * 0.4)
    let mic_radius = approximate_mic_radius(&working_poly);
    let dist = clip_limit.min(mic_radius * 0.4);
    let limit = match working_poly.buffer(-dist, 8) {
        Ok(l) if !l.is_empty().unwrap_or(true) => l,
        _ => Clone::clone(&working_poly),
    };

    // 7. Group ridges by line pair and construct edgelines
    let mut edgelines = build_edgelines(&ridges, &limit, lines);

    // Remove empty edgelines
    edgelines.retain(|e| !e.is_empty().unwrap_or(true) && e.length().unwrap_or(0.0) > 0.0);

    if edgelines.is_empty() {
        return (vec![], vec![]);
    }

    // 9. Snapping
    let mut splitters = Vec::new();
    let mut to_add = Vec::new();

    match snap_to {
        None => {
            // Snap to polygon boundary via shortest line
            if let Some(union_geom) = union_all(&edgelines) {
                if let Ok(boundary) = union_geom.boundary() {
                    if let Ok(poly_boundary) = working_poly.boundary() {
                        if let Some(sl) = make_shortest_line(&boundary, &poly_boundary) {
                            if let Ok(pt) = sl.get_end_point() {
                                splitters.push(pt);
                            }
                            to_add.push(sl);
                        }
                    }
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

    // 10. Simplify to reduce point density
    edgelines = edgelines
        .iter()
        .filter_map(|e| e.simplify(max_segment_length).ok())
        .collect();

    // 11. Line merge and explode
    edgelines = edgelines
        .iter()
        .filter_map(|e| e.line_merge().ok())
        .collect();
    edgelines = explode_multi(&edgelines);

    // Filter out empty/zero-length results
    edgelines.retain(|e| !e.is_empty().unwrap_or(true) && e.length().unwrap_or(0.0) > 0.0);

    (edgelines, splitters)
}

/// Segmentize a geometry: add vertices so no segment exceeds max_length.
pub fn segmentize(geom: &GGeometry, max_length: f64) -> Option<GGeometry> {
    let cs = geom.get_coord_seq().ok()?;
    let n = cs.size().ok()?;
    if n < 2 {
        return Some(Clone::clone(geom));
    }

    let mut new_coords: Vec<[f64; 2]> = Vec::new();

    for i in 0..n - 1 {
        let (x0, y0) = (cs.get_x(i).ok()?, cs.get_y(i).ok()?);
        let (x1, y1) = (cs.get_x(i + 1).ok()?, cs.get_y(i + 1).ok()?);

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
    let (x, y) = (cs.get_x(n - 1).ok()?, cs.get_y(n - 1).ok()?);
    new_coords.push([x, y]);

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
    // Remove ALL copies of any duplicated coordinate, matching Python behavior.
    // Python: `mask = np.isin(points, unq[count > 1]).all(axis=1)`
    // This removes both copies of shared endpoints, creating gaps in the point
    // set that the Voronoi diagram naturally handles.
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

/// A Voronoi ridge: the dual of a Delaunay edge.
/// `line_a` and `line_b` are the source line IDs of the two input points.
/// `vertex_a` and `vertex_b` are the circumcenters of the two adjacent triangles
/// (i.e., the Voronoi edge endpoints).
struct Ridge {
    line_a: usize,
    line_b: usize,
    vertex_a: [f64; 2],
    vertex_b: [f64; 2],
}

/// Extract Voronoi ridges from a Delaunay triangulation.
///
/// Each interior Delaunay edge has two adjacent triangles whose circumcenters
/// form a Voronoi edge. We filter to keep only ridges between different input
/// lines (excluding buffer ridges).
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
        if opp == delaunator::EMPTY {
            continue; // Hull edge — no adjacent triangle
        }
        if e > opp {
            continue; // Process each edge pair once
        }

        let p_a = tri.triangles[e];
        let p_b = tri.triangles[next_halfedge(e)];

        // Skip if both points come from the same line
        let line_a = point_ids[p_a];
        let line_b = point_ids[p_b];
        if line_a == line_b {
            continue;
        }
        // Skip if either is buffer
        if line_a == buffer_line_id || line_b == buffer_line_id {
            continue;
        }

        // Normalize edge key to avoid duplicates
        let edge_key = if p_a < p_b { (p_a, p_b) } else { (p_b, p_a) };
        if !seen_edges.insert(edge_key) {
            continue;
        }

        // Compute circumcenters of the two adjacent triangles
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

        ridges.push(Ridge {
            line_a,
            line_b,
            vertex_a: cc1,
            vertex_b: cc2,
        });
    }

    ridges
}

/// Get next half-edge index in a triangle.
fn next_halfedge(e: usize) -> usize {
    if e % 3 == 2 { e - 2 } else { e + 1 }
}

/// Compute circumcenter of a triangle defined by three points.
fn circumcenter(a: &delaunator::Point, b: &delaunator::Point, c: &delaunator::Point) -> [f64; 2] {
    let d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if d.abs() < 1e-20 {
        // Degenerate triangle — return centroid
        return [(a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0];
    }
    let ux = ((a.x * a.x + a.y * a.y) * (b.y - c.y)
        + (b.x * b.x + b.y * b.y) * (c.y - a.y)
        + (c.x * c.x + c.y * c.y) * (a.y - b.y))
        / d;
    let uy = ((a.x * a.x + a.y * a.y) * (c.x - b.x)
        + (b.x * b.x + b.y * b.y) * (a.x - c.x)
        + (c.x * c.x + c.y * c.y) * (b.x - a.x))
        / d;
    [ux, uy]
}

/// Approximate the maximum inscribed circle radius of a polygon.
///
/// Uses the centroid-to-boundary distance as a reasonable approximation.
fn approximate_mic_radius(poly: &GGeometry) -> f64 {
    let centroid = match poly.get_centroid() {
        Ok(c) => c,
        Err(_) => return 1.0,
    };
    let boundary = match poly.boundary() {
        Ok(b) => b,
        Err(_) => return 1.0,
    };
    centroid.distance(&boundary).unwrap_or(1.0)
}

/// Group ridges by their line pair and construct merged edgelines.
///
/// For each unique pair of input lines (a, b), collect all ridge segments,
/// create LineStrings, merge them with line_merge, clip to the polygon,
/// and handle shared vertex connections.
fn build_edgelines(
    ridges: &[Ridge],
    limit: &GGeometry,
    lines: &[GGeometry],
) -> Vec<GGeometry> {
    let mut edgelines = Vec::new();

    // Group ridges by (line_a, line_b) pair (normalized so a < b)
    let mut ridge_groups: BTreeMap<(usize, usize), Vec<&Ridge>> = BTreeMap::new();
    for ridge in ridges {
        let key = if ridge.line_a < ridge.line_b {
            (ridge.line_a, ridge.line_b)
        } else {
            (ridge.line_b, ridge.line_a)
        };
        ridge_groups.entry(key).or_default().push(ridge);
    }

    // For each line pair, build merged edgeline
    for ((line_a, line_b), group) in &ridge_groups {
        // Create 2-point LineStrings from each ridge's circumcenter pair
        let mut segments: Vec<GGeometry> = Vec::new();
        for ridge in group {
            let wkt = format!(
                "LINESTRING ({} {}, {} {})",
                ridge.vertex_a[0], ridge.vertex_a[1], ridge.vertex_b[0], ridge.vertex_b[1]
            );
            if let Ok(seg) = GGeometry::new_from_wkt(&wkt) {
                if seg.length().unwrap_or(0.0) > 0.0 {
                    segments.push(seg);
                }
            }
        }

        if segments.is_empty() {
            continue;
        }

        // Build a MultiLineString from all ridge segments and line_merge.
        // This matches Python's shapely.line_merge(shapely.multilinestrings(verts)).
        let edgeline = if segments.len() == 1 {
            Clone::clone(&segments[0])
        } else {
            // Build a GEOMETRYCOLLECTION / MULTILINESTRING from segments
            let multi_wkt = format!(
                "MULTILINESTRING ({})",
                segments
                    .iter()
                    .filter_map(|s| {
                        let cs = s.get_coord_seq().ok()?;
                        let x0 = cs.get_x(0).ok()?;
                        let y0 = cs.get_y(0).ok()?;
                        let x1 = cs.get_x(1).ok()?;
                        let y1 = cs.get_y(1).ok()?;
                        Some(format!("({} {}, {} {})", x0, y0, x1, y1))
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let multi = match GGeometry::new_from_wkt(&multi_wkt) {
                Ok(m) => m,
                Err(_) => continue,
            };
            multi.line_merge().unwrap_or(multi)
        };

        // Clip to limit polygon
        let edgeline = clip_edgeline(&edgeline, limit);
        if edgeline.is_empty().unwrap_or(true) {
            continue;
        }

        // Check if the two input lines share a vertex
        if *line_a < lines.len() && *line_b < lines.len() {
            let edgeline = add_shared_vertex_connections(
                edgeline, &lines[*line_a], &lines[*line_b], lines.len(),
            );
            edgelines.push(edgeline);
        } else {
            edgelines.push(edgeline);
        }
    }

    edgelines
}

/// Clip an edgeline to a limit polygon, handling MultiLineStrings.
fn clip_edgeline(edgeline: &GGeometry, limit: &GGeometry) -> GGeometry {
    if edgeline.within(limit).unwrap_or(false) {
        return Clone::clone(edgeline);
    }

    let geom_type = edgeline.geometry_type();
    if geom_type == GeometryTypes::MultiLineString {
        // Clip each part independently
        let n = edgeline.get_num_geometries().unwrap_or(0);
        let mut parts = Vec::new();
        for i in 0..n {
            if let Ok(part) = edgeline.get_geometry_n(i) {
                let owned: GGeometry = Geom::clone(&part);
                if let Ok(clipped) = owned.intersection(limit) {
                    let clipped = remove_sliver(clipped);
                    if !clipped.is_empty().unwrap_or(true) {
                        parts.push(clipped);
                    }
                }
            }
        }
        if parts.is_empty() {
            GGeometry::new_from_wkt("LINESTRING EMPTY").unwrap()
        } else if parts.len() == 1 {
            parts.remove(0)
        } else {
            union_all(&parts).unwrap_or_else(|| GGeometry::new_from_wkt("LINESTRING EMPTY").unwrap())
        }
    } else {
        // Single LineString: clip and remove slivers
        match edgeline.intersection(limit) {
            Ok(clipped) => remove_sliver(clipped),
            Err(_) => Clone::clone(edgeline),
        }
    }
}

/// Remove sliver parts from a MultiLineString (keep only the longest part).
fn remove_sliver(geom: GGeometry) -> GGeometry {
    let geom_type = geom.geometry_type();
    if geom_type == GeometryTypes::MultiLineString {
        let n = geom.get_num_geometries().unwrap_or(0);
        let mut best: Option<GGeometry> = None;
        let mut best_len = 0.0;
        for i in 0..n {
            if let Ok(part) = geom.get_geometry_n(i) {
                let owned: GGeometry = Geom::clone(&part);
                let len = owned.length().unwrap_or(0.0);
                if len > best_len {
                    best_len = len;
                    best = Some(owned);
                }
            }
        }
        best.unwrap_or(geom)
    } else {
        geom
    }
}

/// If two input lines share a vertex, add a shortest line from the shared
/// vertex to the edgeline boundary. This ensures topological connectivity.
fn add_shared_vertex_connections(
    edgeline: GGeometry,
    line_a: &GGeometry,
    line_b: &GGeometry,
    n_lines: usize,
) -> GGeometry {
    let intersection = match line_a.intersection(line_b) {
        Ok(i) => i,
        Err(_) => return edgeline,
    };

    if intersection.is_empty().unwrap_or(true) {
        return edgeline;
    }

    // Skip if MultiPoint with 2 points and more than 2 lines (avoid inner loops)
    let geom_type = intersection.geometry_type();
    if geom_type == GeometryTypes::MultiPoint {
        let n_pts = intersection.get_num_geometries().unwrap_or(0);
        if n_pts == 2 && n_lines != 2 {
            return edgeline;
        }
    }

    // Add shortest lines from intersection points to edgeline boundary
    let edgeline_boundary = match edgeline.boundary() {
        Ok(b) => b,
        Err(_) => return edgeline,
    };

    let n_geoms = if geom_type == GeometryTypes::Point {
        1
    } else {
        intersection.get_num_geometries().unwrap_or(0)
    };

    let mut additions = Vec::new();
    for i in 0..n_geoms {
        let pt = if geom_type == GeometryTypes::Point {
            Clone::clone(&intersection)
        } else if let Ok(g) = intersection.get_geometry_n(i) {
            Geom::clone(&g)
        } else {
            continue;
        };
        if let Some(sl) = make_shortest_line(&pt, &edgeline_boundary) {
            additions.push(sl);
        }
    }

    if additions.is_empty() {
        return edgeline;
    }

    // Union edgeline with all additions
    let mut result = edgeline;
    for add in additions {
        result = match result.union(&add) {
            Ok(u) => u,
            Err(_) => result,
        };
    }
    result
}

/// Explode MultiLineStrings into constituent LineString parts.
fn explode_multi(geoms: &[GGeometry]) -> Vec<GGeometry> {
    let mut result = Vec::new();
    for geom in geoms {
        let geom_type = geom.geometry_type();
        if geom_type == GeometryTypes::MultiLineString || geom_type == GeometryTypes::GeometryCollection {
            let n = geom.get_num_geometries().unwrap_or(0);
            for i in 0..n {
                if let Ok(part) = geom.get_geometry_n(i) {
                    let owned: GGeometry = Geom::clone(&part);
                    if owned.geometry_type() == GeometryTypes::LineString {
                        result.push(owned);
                    }
                }
            }
        } else if geom_type == GeometryTypes::LineString {
            result.push(Clone::clone(geom));
        }
    }
    result
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
