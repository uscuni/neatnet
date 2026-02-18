//! Custom geometry operations that have no direct `geo` crate equivalent.
//!
//! - `polygonize` – half-edge face tracing (replaces GEOS `polygonize_full`)
//! - `line_merge` – merge touching LineStrings at degree-2 nodes
//! - `snap_coords` – snap vertices to nearby target coordinates
//! - `normalize_linestring` – canonical direction for deduplication
//! - `nearest_points` – closest point pair between two geometries
//! - `explode_multi` – flatten Multi*/Collection into simple geometries

use std::collections::{BTreeMap, HashMap, HashSet};

use geo_types::{Coord, LineString, MultiLineString, Polygon};

// ─── Coordinate key helpers ─────────────────────────────────────────────────

/// Coordinate key for exact-float hashing (bit-level equality).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct CoordKey {
    x: i64,
    y: i64,
}

fn coord_key(c: Coord<f64>) -> CoordKey {
    CoordKey {
        x: (c.x * 1e8).round() as i64,
        y: (c.y * 1e8).round() as i64,
    }
}

fn coord_from_key(k: CoordKey) -> Coord<f64> {
    Coord {
        x: k.x as f64 / 1e8,
        y: k.y as f64 / 1e8,
    }
}

// ─── polygonize ─────────────────────────────────────────────────────────────

/// Polygonize a set of LineStrings into face polygons using half-edge tracing.
///
/// Replaces GEOS `polygonize_full()`. Returns the valid interior polygons
/// (CCW-wound faces).
pub fn polygonize(lines: &[LineString<f64>]) -> Vec<Polygon<f64>> {
    // 1. Build directed graph: each undirected edge (u,v) → directed u→v and v→u
    //    Nodes are identified by CoordKey.
    let mut adj: BTreeMap<CoordKey, Vec<CoordKey>> = BTreeMap::new();

    for line in lines {
        let coords = &line.0;
        for w in coords.windows(2) {
            let a = coord_key(w[0]);
            let b = coord_key(w[1]);
            if a == b {
                continue;
            }
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
    }

    // Remove duplicate edges at each node
    for targets in adj.values_mut() {
        targets.sort();
        targets.dedup();
    }

    // 2. Sort outgoing edges by angle at each node
    for (&node, targets) in adj.iter_mut() {
        let nc = coord_from_key(node);
        targets.sort_by(|a, b| {
            let ac = coord_from_key(*a);
            let bc = coord_from_key(*b);
            let angle_a = (ac.y - nc.y).atan2(ac.x - nc.x);
            let angle_b = (bc.y - nc.y).atan2(bc.x - nc.x);
            angle_a.partial_cmp(&angle_b).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // 3. Build "next" map: next(u→v) = at v, from incoming u, take next CW outgoing edge
    //    In a planar graph with edges sorted CCW, the "next" half-edge after u→v
    //    is the one that comes *after* the reverse edge v→u in v's sorted adjacency list.
    let mut next_map: BTreeMap<(CoordKey, CoordKey), (CoordKey, CoordKey)> = BTreeMap::new();

    for (&v, neighbors) in &adj {
        let n = neighbors.len();
        if n == 0 {
            continue;
        }
        for (i, &u) in neighbors.iter().enumerate() {
            // The edge v→u has reverse u→v arriving at v.
            // In CCW-sorted adjacency, the next face half-edge is the
            // PREVIOUS neighbor (clockwise turn), not the next.
            let next_out_idx = (i + n - 1) % n;
            let w = neighbors[next_out_idx];
            // half-edge u→v, next is v→w
            next_map.insert((u, v), (v, w));
        }
    }

    // 4. Trace faces by following next pointers
    let mut used: HashSet<(CoordKey, CoordKey)> = HashSet::new();
    let mut polygons: Vec<Polygon<f64>> = Vec::new();

    for &(start_a, start_b) in next_map.keys() {
        if used.contains(&(start_a, start_b)) {
            continue;
        }

        let mut ring_keys: Vec<CoordKey> = vec![start_a];
        let mut current = (start_a, start_b);
        let mut valid = true;

        loop {
            if used.contains(&current) {
                // Already traced as part of another ring
                valid = false;
                break;
            }
            used.insert(current);
            ring_keys.push(current.1);

            let next = match next_map.get(&current) {
                Some(&n) => n,
                None => {
                    valid = false;
                    break;
                }
            };

            if next.0 == start_a && next.1 == start_b {
                // Completed the ring
                break;
            }
            current = next;

            if ring_keys.len() > next_map.len() + 2 {
                valid = false;
                break;
            }
        }

        if !valid || ring_keys.len() < 4 {
            continue;
        }

        // 5. Convert to coordinates and check winding
        let coords: Vec<Coord<f64>> = ring_keys.iter().map(|k| coord_from_key(*k)).collect();

        // Compute signed area (shoelace formula)
        let signed_area = signed_ring_area(&coords);

        // CCW rings (positive signed area) are interior faces
        // CW ring (negative) is the unbounded exterior
        if signed_area > 0.0 {
            let ring = LineString::new(coords);
            polygons.push(Polygon::new(ring, vec![]));
        }
    }

    polygons
}

/// Signed area of a closed ring (positive = CCW, negative = CW).
fn signed_ring_area(coords: &[Coord<f64>]) -> f64 {
    let n = coords.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    for i in 0..n - 1 {
        area += coords[i].x * coords[i + 1].y;
        area -= coords[i + 1].x * coords[i].y;
    }
    area / 2.0
}

// ─── line_merge ─────────────────────────────────────────────────────────────

/// Merge touching LineStrings at degree-2 nodes into longer LineStrings.
///
/// Replaces GEOS `line_merge()`.
pub fn line_merge(lines: &[LineString<f64>]) -> Vec<LineString<f64>> {
    if lines.is_empty() {
        return vec![];
    }
    if lines.len() == 1 {
        return lines.to_vec();
    }

    // 1. Build undirected graph from endpoints
    //    node → list of (line_index, is_start_endpoint)
    let mut node_to_lines: HashMap<CoordKey, Vec<(usize, bool)>> = HashMap::new();

    for (i, line) in lines.iter().enumerate() {
        if line.0.len() < 2 {
            continue;
        }
        let start = coord_key(line.0[0]);
        let end = coord_key(*line.0.last().unwrap());
        node_to_lines.entry(start).or_default().push((i, true));
        node_to_lines.entry(end).or_default().push((i, false));
    }

    // 2. Find chains via degree-2 nodes
    let mut used = vec![false; lines.len()];
    let mut result: Vec<LineString<f64>> = Vec::new();

    // Start from degree-1 or degree-3+ nodes first, then handle loops.
    // Sort for deterministic ordering (HashMap iteration is non-deterministic).
    let mut start_nodes: Vec<CoordKey> = node_to_lines
        .iter()
        .filter(|(_, v)| v.len() != 2)
        .map(|(&k, _)| k)
        .collect();
    start_nodes.sort();

    for start_node in &start_nodes {
        let line_refs = &node_to_lines[start_node];
        for &(line_idx, is_start) in line_refs {
            if used[line_idx] {
                continue;
            }
            // Walk the chain
            let chain = walk_chain(line_idx, is_start, lines, &node_to_lines, &mut used);
            if !chain.is_empty() {
                result.push(LineString::new(chain));
            }
        }
    }

    // Handle remaining loops (all degree-2 nodes)
    for i in 0..lines.len() {
        if used[i] {
            continue;
        }
        let chain = walk_chain(i, true, lines, &node_to_lines, &mut used);
        if !chain.is_empty() {
            result.push(LineString::new(chain));
        }
    }

    // Any lines not consumed (isolated, degenerate) get passed through
    for (i, line) in lines.iter().enumerate() {
        if !used[i] {
            result.push(line.clone());
        }
    }

    result
}

/// Walk a chain of lines connected at degree-2 nodes, concatenating coordinates.
fn walk_chain(
    start_line: usize,
    start_at_start: bool,
    lines: &[LineString<f64>],
    node_to_lines: &HashMap<CoordKey, Vec<(usize, bool)>>,
    used: &mut [bool],
) -> Vec<Coord<f64>> {
    let mut coords: Vec<Coord<f64>> = Vec::new();
    let mut current_line = start_line;
    let mut forward = start_at_start;

    loop {
        if used[current_line] {
            break;
        }
        used[current_line] = true;

        let line_coords = &lines[current_line].0;
        if line_coords.len() < 2 {
            break;
        }

        // Append coordinates (possibly reversed)
        if forward {
            if coords.is_empty() {
                coords.extend_from_slice(line_coords);
            } else {
                coords.extend_from_slice(&line_coords[1..]);
            }
        } else {
            if coords.is_empty() {
                coords.extend(line_coords.iter().rev());
            } else {
                coords.extend(line_coords.iter().rev().skip(1));
            }
        }

        // Find the next line at the trailing endpoint
        let tail = coord_key(*coords.last().unwrap());
        let connections = match node_to_lines.get(&tail) {
            Some(c) => c,
            None => break,
        };

        // Only continue through degree-2 nodes
        if connections.len() != 2 {
            break;
        }

        // Find the other line at this node
        let next = connections
            .iter()
            .find(|&&(idx, _)| idx != current_line && !used[idx]);

        match next {
            Some(&(next_idx, is_start)) => {
                current_line = next_idx;
                // If the next line's start is at the shared node, we go forward
                // If the next line's end is at the shared node, we go backward
                forward = is_start;
            }
            None => break,
        }
    }

    coords
}

// ─── snap_coords ────────────────────────────────────────────────────────────

/// Snap vertices of a LineString to nearby target coordinates within tolerance.
///
/// Replaces GEOS `snap()` for the specific case of snapping to point targets.
pub fn snap_coords(
    geom: &LineString<f64>,
    targets: &[Coord<f64>],
    tolerance: f64,
) -> LineString<f64> {
    if targets.is_empty() {
        return geom.clone();
    }

    let tol_sq = tolerance * tolerance;
    let coords: Vec<Coord<f64>> = geom
        .0
        .iter()
        .map(|c| {
            let mut best_dist_sq = tol_sq;
            let mut best_target = *c;
            for t in targets {
                let dx = c.x - t.x;
                let dy = c.y - t.y;
                let d_sq = dx * dx + dy * dy;
                if d_sq < best_dist_sq {
                    best_dist_sq = d_sq;
                    best_target = *t;
                }
            }
            best_target
        })
        .collect();

    LineString::new(coords)
}

// ─── normalize_linestring ───────────────────────────────────────────────────

/// Normalize a LineString to a canonical direction for deduplication.
///
/// Compares first and last coordinate lexicographically; reverses if last < first.
/// Replaces GEOS `normalize()` for LineStrings.
pub fn normalize_linestring(line: &LineString<f64>) -> LineString<f64> {
    let coords = &line.0;
    if coords.len() < 2 {
        return line.clone();
    }
    let first = coords[0];
    let last = coords[coords.len() - 1];

    // Closed ring: rotate so smallest coordinate is first, then pick canonical
    // direction by comparing neighbors. Matches Shapely normalize behavior.
    let is_closed = coords.len() >= 4 && first == last;
    if is_closed {
        let ring_coords = &coords[..coords.len() - 1]; // exclude duplicate closing coord
        let n = ring_coords.len();
        let min_idx = ring_coords
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (a.x, a.y).partial_cmp(&(b.x, b.y)).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Rotate ring to start at min_idx
        let mut rotated = Vec::with_capacity(coords.len());
        for i in 0..n {
            rotated.push(ring_coords[(min_idx + i) % n]);
        }
        rotated.push(rotated[0]); // close the ring

        // Normalize direction: compare the second coord with the second-to-last
        // (neighbors of the start in forward vs reverse direction).
        // Pick the direction where the second coord is lexicographically smaller.
        if n >= 3 {
            let fwd = rotated[1];       // second coord (forward direction)
            let rev = rotated[n - 1];   // second-to-last (reverse direction)
            if (rev.x, rev.y) < (fwd.x, fwd.y) {
                // Reverse: keep first, reverse the middle, keep closing coord
                rotated[1..n].reverse();
            }
        }

        return LineString::new(rotated);
    }

    // Open linestring: reverse if last < first lexicographically
    let should_reverse = (last.x, last.y) < (first.x, first.y);

    if should_reverse {
        let reversed: Vec<Coord<f64>> = coords.iter().rev().copied().collect();
        LineString::new(reversed)
    } else {
        line.clone()
    }
}

// ─── nearest_points ─────────────────────────────────────────────────────────

/// Find the nearest point pair between two LineStrings.
///
/// Returns `Some((point_on_a, point_on_b))` or `None` if either is empty.
/// Replaces GEOS `nearest_points()`.
pub fn nearest_points(
    a: &LineString<f64>,
    b: &LineString<f64>,
) -> Option<(Coord<f64>, Coord<f64>)> {
    if a.0.len() < 2 || b.0.len() < 2 {
        // Handle point-like or empty geometries
        if a.0.is_empty() || b.0.is_empty() {
            return None;
        }
        // At least one point each
        let pa = a.0[0];
        let pb = b.0[0];
        return Some((pa, pb));
    }

    let mut best_dist_sq = f64::INFINITY;
    let mut best_a = a.0[0];
    let mut best_b = b.0[0];

    for seg_a in a.0.windows(2) {
        for seg_b in b.0.windows(2) {
            let (pa, pb, d_sq) = nearest_point_segment_segment(
                seg_a[0], seg_a[1], seg_b[0], seg_b[1],
            );
            if d_sq < best_dist_sq {
                best_dist_sq = d_sq;
                best_a = pa;
                best_b = pb;
            }
        }
    }

    Some((best_a, best_b))
}

/// Find nearest points between two line segments (p1-p2) and (p3-p4).
/// Returns (point_on_seg1, point_on_seg2, squared_distance).
fn nearest_point_segment_segment(
    p1: Coord<f64>,
    p2: Coord<f64>,
    p3: Coord<f64>,
    p4: Coord<f64>,
) -> (Coord<f64>, Coord<f64>, f64) {
    // Try all four point-to-segment projections and take the best
    let candidates = [
        project_point_to_segment(p1, p3, p4),
        project_point_to_segment(p2, p3, p4),
        project_point_to_segment(p3, p1, p2),
        project_point_to_segment(p4, p1, p2),
    ];

    let mut best_dist_sq = f64::INFINITY;
    let mut best_a = p1;
    let mut best_b = p3;

    // For candidates 0,1: point is on seg a, projected is on seg b
    let (proj_1, d1_sq) = candidates[0];
    if d1_sq < best_dist_sq {
        best_dist_sq = d1_sq;
        best_a = p1;
        best_b = proj_1;
    }
    let (proj_2, d2_sq) = candidates[1];
    if d2_sq < best_dist_sq {
        best_dist_sq = d2_sq;
        best_a = p2;
        best_b = proj_2;
    }
    // For candidates 2,3: point is on seg b, projected is on seg a
    let (proj_3, d3_sq) = candidates[2];
    if d3_sq < best_dist_sq {
        best_dist_sq = d3_sq;
        best_a = proj_3;
        best_b = p3;
    }
    let (proj_4, d4_sq) = candidates[3];
    if d4_sq < best_dist_sq {
        best_dist_sq = d4_sq;
        best_a = proj_4;
        best_b = p4;
    }

    (best_a, best_b, best_dist_sq)
}

/// Project a point onto a line segment, returning (projected_point, squared_distance).
fn project_point_to_segment(
    p: Coord<f64>,
    a: Coord<f64>,
    b: Coord<f64>,
) -> (Coord<f64>, f64) {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len_sq = dx * dx + dy * dy;

    if len_sq < 1e-20 {
        let d = (p.x - a.x).powi(2) + (p.y - a.y).powi(2);
        return (a, d);
    }

    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj = Coord {
        x: a.x + t * dx,
        y: a.y + t * dy,
    };
    let d = (p.x - proj.x).powi(2) + (p.y - proj.y).powi(2);
    (proj, d)
}

// ─── explode_multi ──────────────────────────────────────────────────────────

/// Explode a MultiLineString into individual LineStrings.
pub fn explode_multi_linestring(mls: &MultiLineString<f64>) -> Vec<LineString<f64>> {
    mls.0.clone()
}

/// Extract LineStrings from a geo_types::Geometry enum (handles Multi* and Collection).
pub fn extract_linestrings(geom: &geo_types::Geometry<f64>) -> Vec<LineString<f64>> {
    match geom {
        geo_types::Geometry::LineString(ls) => vec![ls.clone()],
        geo_types::Geometry::MultiLineString(mls) => mls.0.clone(),
        geo_types::Geometry::GeometryCollection(gc) => {
            gc.0.iter().flat_map(extract_linestrings).collect()
        }
        _ => vec![],
    }
}

/// Extract polygons from a geo_types::Geometry enum.
pub fn extract_polygons(geom: &geo_types::Geometry<f64>) -> Vec<Polygon<f64>> {
    match geom {
        geo_types::Geometry::Polygon(p) => vec![p.clone()],
        geo_types::Geometry::MultiPolygon(mp) => mp.0.clone(),
        geo_types::Geometry::GeometryCollection(gc) => {
            gc.0.iter().flat_map(extract_polygons).collect()
        }
        _ => vec![],
    }
}

// ─── Geometry helpers ───────────────────────────────────────────────────────

/// Compute the bounding box of a LineString as ([min_x, min_y], [max_x, max_y]).
pub fn linestring_bounds(line: &LineString<f64>) -> Option<([f64; 2], [f64; 2])> {
    if line.0.is_empty() {
        return None;
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for c in &line.0 {
        min_x = min_x.min(c.x);
        min_y = min_y.min(c.y);
        max_x = max_x.max(c.x);
        max_y = max_y.max(c.y);
    }
    Some(([min_x, min_y], [max_x, max_y]))
}

/// Compute the bounding box of a Polygon.
pub fn polygon_bounds(poly: &Polygon<f64>) -> Option<([f64; 2], [f64; 2])> {
    linestring_bounds(poly.exterior())
}

// ─── WKT helpers ────────────────────────────────────────────────────────────

/// Format a LineString as WKT (for deduplication/normalization).
pub fn linestring_to_wkt(line: &LineString<f64>) -> String {
    if line.0.is_empty() {
        return "LINESTRING EMPTY".to_string();
    }
    let coords: Vec<String> = line.0.iter().map(|c| format!("{} {}", c.x, c.y)).collect();
    format!("LINESTRING ({})", coords.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ls(coords: &[[f64; 2]]) -> LineString<f64> {
        LineString::new(coords.iter().map(|c| Coord { x: c[0], y: c[1] }).collect())
    }

    #[test]
    fn test_line_merge_chain() {
        let l1 = ls(&[[0.0, 0.0], [1.0, 0.0]]);
        let l2 = ls(&[[1.0, 0.0], [2.0, 0.0]]);
        let l3 = ls(&[[2.0, 0.0], [3.0, 0.0]]);
        let result = line_merge(&[l1, l2, l3]);
        assert_eq!(result.len(), 1, "chain should merge to 1 line");
        assert_eq!(result[0].0.len(), 4);
    }

    #[test]
    fn test_line_merge_disconnected() {
        let l1 = ls(&[[0.0, 0.0], [1.0, 0.0]]);
        let l2 = ls(&[[10.0, 0.0], [11.0, 0.0]]);
        let result = line_merge(&[l1, l2]);
        assert_eq!(result.len(), 2, "disconnected lines stay separate");
    }

    #[test]
    fn test_line_merge_t_junction() {
        let l1 = ls(&[[0.0, 0.0], [1.0, 0.0]]);
        let l2 = ls(&[[1.0, 0.0], [2.0, 0.0]]);
        let l3 = ls(&[[1.0, 0.0], [1.0, 1.0]]);
        let result = line_merge(&[l1, l2, l3]);
        // Node (1,0) has degree 3 → no merge through it
        assert_eq!(result.len(), 3, "T-junction prevents merge");
    }

    #[test]
    fn test_normalize_linestring_already_canonical() {
        let l = ls(&[[0.0, 0.0], [1.0, 0.0]]);
        let n = normalize_linestring(&l);
        assert_eq!(n.0[0], l.0[0]);
    }

    #[test]
    fn test_normalize_linestring_reversed() {
        let l = ls(&[[1.0, 0.0], [0.0, 0.0]]);
        let n = normalize_linestring(&l);
        assert_eq!(n.0[0], Coord { x: 0.0, y: 0.0 });
        assert_eq!(n.0[1], Coord { x: 1.0, y: 0.0 });
    }

    #[test]
    fn test_snap_coords_basic() {
        let l = ls(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        let targets = vec![Coord { x: 0.9, y: 0.1 }];
        let snapped = snap_coords(&l, &targets, 0.2);
        // Middle point (1,0) should snap to (0.9, 0.1) since dist ≈ 0.14 < 0.2
        assert!((snapped.0[1].x - 0.9).abs() < 1e-10);
        assert!((snapped.0[1].y - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_snap_coords_no_snap() {
        let l = ls(&[[0.0, 0.0], [1.0, 0.0]]);
        let targets = vec![Coord { x: 5.0, y: 5.0 }];
        let snapped = snap_coords(&l, &targets, 0.1);
        assert_eq!(snapped.0[0], l.0[0]);
        assert_eq!(snapped.0[1], l.0[1]);
    }

    #[test]
    fn test_nearest_points_basic() {
        let a = ls(&[[0.0, 0.0], [10.0, 0.0]]);
        let b = ls(&[[5.0, 3.0], [5.0, 10.0]]);
        let (pa, pb) = nearest_points(&a, &b).unwrap();
        assert!((pa.x - 5.0).abs() < 1e-10);
        assert!((pa.y - 0.0).abs() < 1e-10);
        assert!((pb.x - 5.0).abs() < 1e-10);
        assert!((pb.y - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygonize_square() {
        // Four edges forming a square
        let l1 = ls(&[[0.0, 0.0], [10.0, 0.0]]);
        let l2 = ls(&[[10.0, 0.0], [10.0, 10.0]]);
        let l3 = ls(&[[10.0, 10.0], [0.0, 10.0]]);
        let l4 = ls(&[[0.0, 10.0], [0.0, 0.0]]);
        let polys = polygonize(&[l1, l2, l3, l4]);
        assert_eq!(polys.len(), 1, "four edges forming a square should produce 1 polygon");
        use geo::Area;
        let area = polys[0].unsigned_area();
        assert!((area - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_polygonize_two_squares() {
        // Two adjacent squares sharing an edge
        let l1 = ls(&[[0.0, 0.0], [10.0, 0.0]]);
        let l2 = ls(&[[10.0, 0.0], [10.0, 10.0]]);
        let l3 = ls(&[[10.0, 10.0], [0.0, 10.0]]);
        let l4 = ls(&[[0.0, 10.0], [0.0, 0.0]]);
        let l5 = ls(&[[10.0, 0.0], [20.0, 0.0]]);
        let l6 = ls(&[[20.0, 0.0], [20.0, 10.0]]);
        let l7 = ls(&[[20.0, 10.0], [10.0, 10.0]]);
        let polys = polygonize(&[l1, l2, l3, l4, l5, l6, l7]);
        assert_eq!(polys.len(), 2, "two adjacent squares should produce 2 polygons");
    }

    #[test]
    fn test_explode_multi_linestring() {
        let mls = MultiLineString::new(vec![
            ls(&[[0.0, 0.0], [1.0, 0.0]]),
            ls(&[[2.0, 0.0], [3.0, 0.0]]),
        ]);
        let result = explode_multi_linestring(&mls);
        assert_eq!(result.len(), 2);
    }
}
