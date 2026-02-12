//! Gap closing and line extension utilities.
//!
//! Ports Python `neatnet.gaps`: `close_gaps`, `extend_lines`.

use std::collections::HashMap;

use geos::{Geom, Geometry as GGeometry};

/// Close gaps in LineString geometry where it should be contiguous.
///
/// Snaps both lines to the centroid of a gap between them.
///
/// Mirrors Python `close_gaps()`.
pub fn close_gaps(
    geometries: &[GGeometry],
    tolerance: f64,
) -> Vec<GGeometry> {
    if geometries.is_empty() {
        return vec![];
    }

    // 1. Extract start/end coordinates
    let mut edge_points: Vec<[f64; 2]> = Vec::new();
    for geom in geometries {
        let Ok(cs) = geom.get_coord_seq() else { continue };
        let Ok(n) = cs.size() else { continue };
        if n < 2 {
            continue;
        }
        if let (Ok(x), Ok(y)) = (cs.get_x(0), cs.get_y(0)) {
            edge_points.push([x, y]);
        }
        if let (Ok(x), Ok(y)) = (cs.get_x(n - 1), cs.get_y(n - 1)) {
            edge_points.push([x, y]);
        }
    }

    // 2. Get unique points
    let unique_points = deduplicate_coords(&edge_points);

    // 3. Buffer points and dissolve
    let half_tol = tolerance / 2.0;
    let mut buffer_geoms: Vec<GGeometry> = Vec::new();
    for p in &unique_points {
        let wkt = format!("POINT ({} {})", p[0], p[1]);
        if let Ok(pt) = GGeometry::new_from_wkt(&wkt) {
            if let Ok(buf) = pt.buffer(half_tol, 8) {
                buffer_geoms.push(buf);
            }
        }
    }

    if buffer_geoms.is_empty() {
        return geometries.to_vec();
    }

    // Union all buffers
    let mut dissolved = Clone::clone(&buffer_geoms[0]);
    for buf in &buffer_geoms[1..] {
        if let Ok(u) = dissolved.union(buf) {
            dissolved = u;
        }
    }

    // 4. Get centroids of each resulting polygon
    let mut centroids: Vec<GGeometry> = Vec::new();
    let n_parts = dissolved.get_num_geometries().unwrap_or(1);
    if n_parts > 1 {
        for i in 0..n_parts {
            if let Ok(part) = dissolved.get_geometry_n(i) {
                if let Ok(centroid) = part.get_centroid() {
                    centroids.push(centroid);
                }
            }
        }
    } else {
        if let Ok(centroid) = dissolved.get_centroid() {
            centroids.push(centroid);
        }
    }

    // 5. Union centroids and snap geometries to them
    if centroids.is_empty() {
        return geometries.to_vec();
    }

    let mut centroid_union = Clone::clone(&centroids[0]);
    for c in &centroids[1..] {
        if let Ok(u) = centroid_union.union(c) {
            centroid_union = u;
        }
    }

    // Snap each geometry to the centroid union
    let mut result = Vec::with_capacity(geometries.len());
    for geom in geometries {
        match geom.snap(&centroid_union, tolerance) {
            Ok(snapped) => result.push(snapped),
            Err(_) => result.push(Clone::clone(geom)),
        }
    }

    result
}

/// Extend lines to connect to nearby features within tolerance.
///
/// Extends unjoined ends (degree-1 nodes) of LineString segments to
/// join with other segments or a target geometry.
///
/// Mirrors Python `extend_lines()`.
pub fn extend_lines(
    geometries: &[GGeometry],
    tolerance: f64,
    target: Option<&[GGeometry]>,
    _barrier: Option<&[GGeometry]>,
    _extension: f64,
) -> Vec<GGeometry> {
    if geometries.is_empty() {
        return vec![];
    }

    let target_geoms = target.unwrap_or(geometries);
    let _is_self_target = target.is_none();

    // Build spatial index of endpoints to find degree-1 nodes
    let mut endpoint_counts: HashMap<CoordKey, Vec<usize>> = HashMap::new();
    for (i, geom) in geometries.iter().enumerate() {
        let Ok(cs) = geom.get_coord_seq() else { continue };
        let Ok(n) = cs.size() else { continue };
        if n < 2 {
            continue;
        }
        if let (Ok(x), Ok(y)) = (cs.get_x(0), cs.get_y(0)) {
            endpoint_counts
                .entry(coord_key(x, y))
                .or_default()
                .push(i);
        }
        if let (Ok(x), Ok(y)) = (cs.get_x(n - 1), cs.get_y(n - 1)) {
            endpoint_counts
                .entry(coord_key(x, y))
                .or_default()
                .push(i);
        }
    }

    // Identify degree-1 endpoints (only touched by one geometry)
    let degree1_edges: Vec<usize> = endpoint_counts
        .values()
        .filter(|v| v.len() == 1)
        .flat_map(|v| v.iter().copied())
        .collect();

    let mut result = geometries.to_vec();

    for &edge_idx in &degree1_edges {
        let geom = &geometries[edge_idx];
        let Ok(cs) = geom.get_coord_seq() else { continue };
        let Ok(n) = cs.size() else { continue };
        if n < 2 {
            continue;
        }

        // Check which end(s) need extension
        let (Ok(sx), Ok(sy)) = (cs.get_x(0), cs.get_y(0)) else { continue };
        let (Ok(ex), Ok(ey)) = (cs.get_x(n - 1), cs.get_y(n - 1)) else { continue };

        let start_key = coord_key(sx, sy);
        let end_key = coord_key(ex, ey);

        let start_deg1 = endpoint_counts
            .get(&start_key)
            .map(|v| v.len() == 1)
            .unwrap_or(false);
        let end_deg1 = endpoint_counts
            .get(&end_key)
            .map(|v| v.len() == 1)
            .unwrap_or(false);

        // Try to extend the line using the extrapolation logic
        if start_deg1 && !end_deg1 {
            // Extend from start (reverse direction)
            if let Some(extended) = try_extend_from_end(geom, target_geoms, tolerance, true) {
                result[edge_idx] = extended;
            }
        } else if !start_deg1 && end_deg1 {
            // Extend from end
            if let Some(extended) = try_extend_from_end(geom, target_geoms, tolerance, false) {
                result[edge_idx] = extended;
            }
        }
    }

    result
}

// ─── Internal helpers ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CoordKey {
    x: i64,
    y: i64,
}

fn coord_key(x: f64, y: f64) -> CoordKey {
    CoordKey {
        x: (x * 1e8).round() as i64,
        y: (y * 1e8).round() as i64,
    }
}

fn deduplicate_coords(coords: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for c in coords {
        let key = ((c[0] * 1e8).round() as i64, (c[1] * 1e8).round() as i64);
        if seen.insert(key) {
            unique.push(*c);
        }
    }
    unique
}

/// Try to extend a line from one end to the nearest target.
fn try_extend_from_end(
    geom: &GGeometry,
    targets: &[GGeometry],
    tolerance: f64,
    from_start: bool,
) -> Option<GGeometry> {
    let cs = geom.get_coord_seq().ok()?;
    let n = cs.size().ok()?;
    if n < 2 {
        return None;
    }

    // Get the last two points for direction
    let (p1_x, p1_y, p2_x, p2_y) = if from_start {
        (
            cs.get_x(1).ok()?,
            cs.get_y(1).ok()?,
            cs.get_x(0).ok()?,
            cs.get_y(0).ok()?,
        )
    } else {
        (
            cs.get_x(n - 2).ok()?,
            cs.get_y(n - 2).ok()?,
            cs.get_x(n - 1).ok()?,
            cs.get_y(n - 1).ok()?,
        )
    };

    // Extrapolate in the p1→p2 direction
    let dx = p2_x - p1_x;
    let dy = p2_y - p1_y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-12 {
        return None;
    }

    let ext_x = p2_x + dx / len * tolerance;
    let ext_y = p2_y + dy / len * tolerance;

    // Create extrapolation line
    let ext_wkt = format!("LINESTRING ({} {}, {} {})", p2_x, p2_y, ext_x, ext_y);
    let ext_line = GGeometry::new_from_wkt(&ext_wkt).ok()?;

    // Find intersection with targets
    for target in targets {
        if let Ok(inter) = ext_line.intersection(target) {
            if !inter.is_empty().unwrap_or(true) {
                // Get the intersection point
                if let Ok(pt_cs) = inter.get_coord_seq() {
                    if pt_cs.size().ok()? > 0 {
                        let new_x = pt_cs.get_x(0).ok()?;
                        let new_y = pt_cs.get_y(0).ok()?;

                        // Build new geometry with the extended point
                        let mut coords: Vec<[f64; 2]> = Vec::new();
                        if from_start {
                            coords.push([new_x, new_y]);
                        }
                        for i in 0..n {
                            coords.push([cs.get_x(i).ok()?, cs.get_y(i).ok()?]);
                        }
                        if !from_start {
                            coords.push([new_x, new_y]);
                        }

                        let wkt = format!(
                            "LINESTRING ({})",
                            coords
                                .iter()
                                .map(|c| format!("{} {}", c[0], c[1]))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                        return GGeometry::new_from_wkt(&wkt).ok();
                    }
                }
            }
        }
    }

    None
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
    fn test_close_gaps_no_gaps() {
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let result = close_gaps(&[g1, g2], 0.5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_deduplicate_coords() {
        let coords = vec![[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]];
        let unique = deduplicate_coords(&coords);
        assert_eq!(unique.len(), 2);
    }
}
