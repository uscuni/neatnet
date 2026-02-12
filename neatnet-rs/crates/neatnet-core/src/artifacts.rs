//! Face artifact detection and classification.
//!
//! Ports Python `neatnet.artifacts`: `FaceArtifacts`, `get_artifacts`,
//! and all the artifact processing functions (n1_g1, nx_gx, etc.).

use std::f64::consts::PI;

use geos::{Geom, Geometry as GGeometry, GeometryTypes};
use petgraph::graph::UnGraph;

use crate::spatial;

/// Shape metric: isoareal quotient (Altman's PA_3).
///
/// Ratio of the perimeter of an equal-area circle to the polygon's perimeter.
/// Formula: `2π√(A/π) / P`. Ranges from 0 (elongated) to 1 (circular).
///
/// Matches Python `esda.shape.isoareal_quotient`.
pub fn isoareal_quotient(area: f64, perimeter: f64) -> f64 {
    if area <= 0.0 || perimeter <= 0.0 {
        return 0.0;
    }
    2.0 * PI * (area / PI).sqrt() / perimeter
}

/// Shape metric: isoperimetric quotient (Altman's PA_1).
///
/// Ratio of the polygon's area to the area of an equal-perimeter circle.
/// Formula: `4πA/P²`. Ranges from 0 (elongated) to 1 (circular).
///
/// Matches Python `esda.shape.isoperimetric_quotient`.
pub fn isoperimetric_quotient(area: f64, perimeter: f64) -> f64 {
    if perimeter <= 0.0 {
        return 0.0;
    }
    4.0 * PI * area / (perimeter * perimeter)
}

/// Minimum bounding circle ratio (Reock compactness).
///
/// Computes `area / (π * r²)` where `r` is the minimum bounding circle radius.
/// Uses the exact minimum enclosing circle from convex hull vertices.
///
/// Mirrors Python `esda.shape.minimum_bounding_circle_ratio()`.
pub fn minimum_bounding_circle_ratio(geom: &GGeometry) -> f64 {
    let area = geom.area().unwrap_or(0.0);
    if area <= 0.0 {
        return 0.0;
    }

    // Get convex hull vertices
    let hull = match geom.convex_hull() {
        Ok(h) => h,
        Err(_) => return 0.0,
    };
    let ring = match hull.get_exterior_ring() {
        Ok(r) => r,
        Err(_) => return 0.0,
    };
    let cs = match ring.get_coord_seq() {
        Ok(c) => c,
        Err(_) => return 0.0,
    };
    let n = cs.size().unwrap_or(0);
    if n < 3 {
        return 0.0;
    }

    // Extract hull vertex coordinates (skip last = first for closed ring)
    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(n - 1);
    for i in 0..n.saturating_sub(1) {
        if let (Ok(x), Ok(y)) = (cs.get_x(i), cs.get_y(i)) {
            pts.push([x, y]);
        }
    }

    let radius = minimum_enclosing_circle_radius(&pts);
    let mbc_area = std::f64::consts::PI * radius * radius;
    if mbc_area <= 0.0 {
        return 0.0;
    }

    area / mbc_area
}

/// Compute the radius of the minimum enclosing circle for a set of 2D points.
///
/// Uses Welzl's randomized algorithm, O(n) expected time.
fn minimum_enclosing_circle_radius(points: &[[f64; 2]]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    if points.len() == 1 {
        return 0.0;
    }

    // Shuffle for expected O(n) — use deterministic shuffle via indices
    let mut indices: Vec<usize> = (0..points.len()).collect();
    // Simple deterministic shuffle based on coordinate sums
    indices.sort_by(|&a, &b| {
        let sa = points[a][0] + points[a][1] * 1e7;
        let sb = points[b][0] + points[b][1] * 1e7;
        sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut cx = (points[indices[0]][0] + points[indices[1]][0]) / 2.0;
    let mut cy = (points[indices[0]][1] + points[indices[1]][1]) / 2.0;
    let dx = points[indices[0]][0] - points[indices[1]][0];
    let dy = points[indices[0]][1] - points[indices[1]][1];
    let mut r_sq = (dx * dx + dy * dy) / 4.0;

    for i in 2..indices.len() {
        let p = &points[indices[i]];
        let dpx = p[0] - cx;
        let dpy = p[1] - cy;
        if dpx * dpx + dpy * dpy <= r_sq * (1.0 + 1e-10) {
            continue; // point inside current circle
        }

        // Point outside — find min circle with p on boundary
        cx = (points[indices[0]][0] + p[0]) / 2.0;
        cy = (points[indices[0]][1] + p[1]) / 2.0;
        let dx = points[indices[0]][0] - p[0];
        let dy = points[indices[0]][1] - p[1];
        r_sq = (dx * dx + dy * dy) / 4.0;

        for j in 1..i {
            let q = &points[indices[j]];
            let dqx = q[0] - cx;
            let dqy = q[1] - cy;
            if dqx * dqx + dqy * dqy <= r_sq * (1.0 + 1e-10) {
                continue; // q inside current circle
            }

            // Circle through p and q
            cx = (q[0] + p[0]) / 2.0;
            cy = (q[1] + p[1]) / 2.0;
            let dx = q[0] - p[0];
            let dy = q[1] - p[1];
            r_sq = (dx * dx + dy * dy) / 4.0;

            for k in 0..j {
                let s = &points[indices[k]];
                let dsx = s[0] - cx;
                let dsy = s[1] - cy;
                if dsx * dsx + dsy * dsy <= r_sq * (1.0 + 1e-10) {
                    continue; // s inside current circle
                }

                // Circle through p, q, s (circumcircle of triangle)
                if let Some((ccx, ccy, cr_sq)) = circumcircle(p, q, s) {
                    cx = ccx;
                    cy = ccy;
                    r_sq = cr_sq;
                }
            }
        }
    }

    r_sq.sqrt()
}

/// Compute the circumcircle (center and radius²) of three points.
fn circumcircle(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> Option<(f64, f64, f64)> {
    let ax = a[0];
    let ay = a[1];
    let bx = b[0];
    let by = b[1];
    let cx = c[0];
    let cy = c[1];

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-14 {
        return None; // collinear
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    let dx = ax - ux;
    let dy = ay - uy;
    Some((ux, uy, dx * dx + dy * dy))
}

/// Gaussian Kernel Density Estimation with Silverman bandwidth.
///
/// Returns (grid, pdf) evaluated at `n_points` equally spaced values
/// between `min_val` and `max_val`.
pub fn gaussian_kde(
    data: &[f64],
    n_points: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n = data.len() as f64;
    if data.is_empty() {
        return (vec![], vec![]);
    }

    // Silverman bandwidth
    let mean: f64 = data.iter().sum::<f64>() / n;
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // IQR-based bandwidth
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = sorted[(n as usize) / 4];
    let q3 = sorted[(3 * n as usize) / 4];
    let iqr = q3 - q1;

    let silverman = 0.9 * std_dev.min(iqr / 1.34) * n.powf(-0.2);
    let bandwidth = if silverman > 0.0 { silverman } else { std_dev * 0.5 };

    let min_val = sorted[0];
    let max_val = sorted[sorted.len() - 1];

    let grid: Vec<f64> = (0..n_points)
        .map(|i| min_val + (max_val - min_val) * i as f64 / (n_points - 1) as f64)
        .collect();

    let pdf: Vec<f64> = grid
        .iter()
        .map(|&x| {
            let sum: f64 = data
                .iter()
                .map(|&xi| {
                    let z = (x - xi) / bandwidth;
                    (-0.5 * z * z).exp()
                })
                .sum();
            sum / (n * bandwidth * (2.0 * PI).sqrt())
        })
        .collect();

    (grid, pdf)
}

/// Find peaks in a 1-D signal with minimum height and prominence.
///
/// Simplified version of `scipy.signal.find_peaks`.
pub fn find_peaks(
    signal: &[f64],
    min_height: Option<f64>,
    min_prominence: Option<f64>,
) -> Vec<usize> {
    let n = signal.len();
    if n < 3 {
        return vec![];
    }

    // Find local maxima
    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..n - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peaks.push(i);
        }
    }

    // Filter by height
    if let Some(height) = min_height {
        peaks.retain(|&i| signal[i] >= height);
    }

    // Filter by prominence
    if let Some(prominence) = min_prominence {
        peaks.retain(|&peak| {
            let prom = compute_prominence(signal, peak);
            prom >= prominence
        });
    }

    peaks
}

/// Compute the prominence of a peak in a signal.
fn compute_prominence(signal: &[f64], peak: usize) -> f64 {
    let peak_val = signal[peak];

    // Search left for the lowest valley before a higher peak
    let mut left_min = peak_val;
    for i in (0..peak).rev() {
        left_min = left_min.min(signal[i]);
        if signal[i] > peak_val {
            break;
        }
    }

    // Search right for the lowest valley before a higher peak
    let mut right_min = peak_val;
    for i in (peak + 1)..signal.len() {
        right_min = right_min.min(signal[i]);
        if signal[i] > peak_val {
            break;
        }
    }

    peak_val - left_min.max(right_min)
}

/// Polygonize a street network and detect face artifacts.
///
/// Returns the FAI threshold and artifact polygons.
///
/// Mirrors Python `FaceArtifacts.__init__()` + `get_artifacts()`.
pub fn detect_artifacts(
    geometries: &[GGeometry],
    threshold: Option<f64>,
    threshold_fallback: f64,
) -> Option<(Vec<GGeometry>, Vec<f64>, f64)> {
    // 1. Union all lines and polygonize
    let union = union_all_lines(geometries)?;
    let polygons = union.polygonize_full().ok()?;
    // polygonize_full returns (valid_polygons, dangles, cuts, invalid)
    // We want the valid polygons

    // 2. Extract individual polygons and compute FAI
    let mut poly_geoms = Vec::new();
    let mut fai_values = Vec::new();

    let n_geoms = polygons.0.get_num_geometries().unwrap_or(0);
    for i in 0..n_geoms {
        if let Ok(poly) = polygons.0.get_geometry_n(i) {
            let owned: GGeometry = Geom::clone(&poly);
            let area = owned.area().unwrap_or(0.0);
            let mbc_ratio = minimum_bounding_circle_ratio(&owned);
            let fai = (mbc_ratio * area).ln();
            fai_values.push(fai);
            poly_geoms.push(owned);
        }
    }

    if poly_geoms.is_empty() {
        return None;
    }

    // 3. Determine threshold via KDE
    let final_threshold = if let Some(t) = threshold {
        t
    } else {
        match find_fai_threshold(&fai_values) {
            Some(t) => t,
            None => threshold_fallback,
        }
    };

    // 4. Filter artifacts below threshold
    let mut artifact_geoms = Vec::new();
    let mut artifact_fais = Vec::new();
    for (geom, &fai) in poly_geoms.iter().zip(fai_values.iter()) {
        if fai < final_threshold {
            artifact_geoms.push(Clone::clone(geom));
            artifact_fais.push(fai);
        }
    }
    Some((artifact_geoms, artifact_fais, final_threshold))
}

/// Full artifact detection with iterative expansion.
///
/// 1. Polygonize the network and compute FAI threshold.
/// 2. Build rook contiguity graph.
/// 3. Iteratively expand artifacts:
///    - Block-like: (enclosed OR touching) AND small AND elongated
///    - Circle-like enclosed: enclosed AND small AND circular
///    - Circle-like touching: touching AND small AND circular
/// 4. Filter by exclusion mask if provided.
///
/// Returns `(artifact_polygons, artifact_fais, threshold)`.
///
/// Mirrors Python `get_artifacts()`.
pub fn get_artifacts(
    geometries: &[GGeometry],
    threshold: Option<f64>,
    threshold_fallback: f64,
    exclusion_mask: Option<&[GGeometry]>,
    area_threshold_blocks: f64,
    isoareal_threshold_blocks: f64,
    area_threshold_circles: f64,
    isoareal_threshold_circles_enclosed: f64,
    isoperimetric_threshold_circles_touching: f64,
) -> Option<(Vec<GGeometry>, Vec<f64>, f64)> {
    // 1. Polygonize and compute initial artifacts
    let union = union_all_lines(geometries)?;
    let polygons = union.polygonize_full().ok()?;

    let mut poly_geoms = Vec::new();
    let mut fai_values = Vec::new();

    let n_geoms = polygons.0.get_num_geometries().unwrap_or(0);
    for i in 0..n_geoms {
        if let Ok(poly) = polygons.0.get_geometry_n(i) {
            let owned: GGeometry = Geom::clone(&poly);
            let area = owned.area().unwrap_or(0.0);
            let mbc_ratio = minimum_bounding_circle_ratio(&owned);
            let fai = (mbc_ratio * area).ln();
            fai_values.push(fai);
            poly_geoms.push(owned);
        }
    }

    if poly_geoms.is_empty() {
        return None;
    }

    // 2. Determine FAI threshold
    let final_threshold = if let Some(t) = threshold {
        t
    } else {
        match find_fai_threshold(&fai_values) {
            Some(t) => t,
            None => threshold_fallback,
        }
    };

    // 3. Initialize is_artifact flags
    let mut is_artifact: Vec<bool> = fai_values.iter().map(|&fai| fai < final_threshold).collect();

    // 4. Pre-compute shape metrics for all polygons
    let areas: Vec<f64> = poly_geoms.iter().map(|g| g.area().unwrap_or(0.0)).collect();
    let perimeters: Vec<f64> = poly_geoms.iter().map(|g| g.length().unwrap_or(0.0)).collect();
    let isoareal: Vec<f64> = areas
        .iter()
        .zip(perimeters.iter())
        .map(|(&a, &p)| isoareal_quotient(a, p))
        .collect();
    let isoperimetric: Vec<f64> = areas
        .iter()
        .zip(perimeters.iter())
        .map(|(&a, &p)| isoperimetric_quotient(a, p))
        .collect();

    // 5. Build rook contiguity graph
    let adjacency = build_contiguity_graph(&poly_geoms, true);

    // Identify isolates (polygons with no neighbors)
    let is_isolate: Vec<bool> = adjacency.iter().map(|adj| adj.is_empty()).collect();

    // 6. Iterative expansion
    loop {
        let artifact_count_before: usize = is_artifact.iter().filter(|&&a| a).count();

        // Compute enclosed/touching from contiguity graph
        let (enclosed, touching) =
            compute_enclosed_touching(&adjacency, &is_artifact, &is_isolate);

        // Block-like: (enclosed OR touching) AND small area AND elongated
        for i in 0..poly_geoms.len() {
            if is_artifact[i] {
                continue;
            }
            if (enclosed[i] || touching[i])
                && areas[i] < area_threshold_blocks
                && isoareal[i] < isoareal_threshold_blocks
            {
                is_artifact[i] = true;
            }
        }

        // Circle-like enclosed: enclosed AND small AND circular
        for i in 0..poly_geoms.len() {
            if is_artifact[i] {
                continue;
            }
            if enclosed[i]
                && areas[i] < area_threshold_circles
                && isoareal[i] > isoareal_threshold_circles_enclosed
            {
                is_artifact[i] = true;
            }
        }

        // Circle-like touching: touching AND small AND circular
        for i in 0..poly_geoms.len() {
            if is_artifact[i] {
                continue;
            }
            if touching[i]
                && areas[i] < area_threshold_circles
                && isoperimetric[i] > isoperimetric_threshold_circles_touching
            {
                is_artifact[i] = true;
            }
        }

        let artifact_count_after: usize = is_artifact.iter().filter(|&&a| a).count();
        if artifact_count_after == artifact_count_before {
            break;
        }
    }

    // 7. Apply exclusion mask if provided
    if let Some(mask) = exclusion_mask {
        for i in 0..poly_geoms.len() {
            if !is_artifact[i] {
                continue;
            }
            for mask_geom in mask {
                if poly_geoms[i].intersects(mask_geom).unwrap_or(false) {
                    is_artifact[i] = false;
                    break;
                }
            }
        }
    }

    // 8. Collect artifact polygons
    let mut artifact_geoms = Vec::new();
    let mut artifact_fais = Vec::new();
    for (i, geom) in poly_geoms.iter().enumerate() {
        if is_artifact[i] {
            artifact_geoms.push(Clone::clone(geom));
            artifact_fais.push(fai_values[i]);
        }
    }

    Some((artifact_geoms, artifact_fais, final_threshold))
}

/// Compute enclosed/touching flags from contiguity graph and artifact flags.
///
/// - `enclosed[i]` = true iff ALL neighbors of i are artifacts
/// - `touching[i]` = true iff ANY neighbor of i is an artifact
///
/// Isolates are always false for both.
fn compute_enclosed_touching(
    adjacency: &[Vec<usize>],
    is_artifact: &[bool],
    is_isolate: &[bool],
) -> (Vec<bool>, Vec<bool>) {
    let n = adjacency.len();
    let mut enclosed = vec![false; n];
    let mut touching = vec![false; n];

    for i in 0..n {
        if is_isolate[i] || is_artifact[i] {
            continue;
        }
        let neighbors = &adjacency[i];
        if neighbors.is_empty() {
            continue;
        }
        let all_artifact = neighbors.iter().all(|&j| is_artifact[j]);
        let any_artifact = neighbors.iter().any(|&j| is_artifact[j]);
        enclosed[i] = all_artifact;
        touching[i] = any_artifact;
    }

    (enclosed, touching)
}

/// Find the FAI threshold from KDE analysis.
///
/// Finds the valley between the two highest peaks.
fn find_fai_threshold(fai_values: &[f64]) -> Option<f64> {
    let (grid, pdf) = gaussian_kde(fai_values, 1000);

    // Find peaks in PDF
    let peaks = find_peaks(&pdf, Some(0.008), Some(0.00075));
    if peaks.len() < 2 {
        return None;
    }

    // Find valleys (peaks in inverted signal)
    let inverted: Vec<f64> = pdf.iter().map(|&v| -v + 1.0).collect();
    let valleys = find_peaks(&inverted, None, Some(0.00075));
    if valleys.is_empty() {
        return None;
    }

    // Find highest peak
    let highest_peak_idx = peaks
        .iter()
        .max_by(|&&a, &&b| pdf[a].partial_cmp(&pdf[b]).unwrap())?;

    // Find valley between the highest peak and its neighbor
    for &valley in &valleys {
        // Check if valley is between two peaks containing the highest peak
        for pair in peaks.windows(2) {
            if pair[0] <= valley && valley <= pair[1] {
                if pair.contains(highest_peak_idx) {
                    return Some(grid[valley]);
                }
            }
        }
    }

    None
}

/// Build a rook contiguity graph from polygons.
///
/// Two polygons are neighbors if they share an edge (not just a point).
pub fn build_contiguity_graph(
    polygons: &[GGeometry],
    rook: bool,
) -> Vec<Vec<usize>> {
    let n = polygons.len();
    let tree = spatial::build_rtree(polygons);
    let mut adjacency: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        let env = match polygons[i].envelope() {
            Ok(e) => e,
            Err(_) => continue,
        };
        // Handle Polygon envelopes (get_coord_seq only works on Point/LineString/Ring)
        let cs = if env.geometry_type() == GeometryTypes::Polygon {
            match env.get_exterior_ring().and_then(|r| r.get_coord_seq()) {
                Ok(c) => c,
                Err(_) => continue,
            }
        } else {
            match env.get_coord_seq() {
                Ok(c) => c,
                Err(_) => continue,
            }
        };

        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for j in 0..cs.size().unwrap_or(0) {
            if let (Ok(x), Ok(y)) = (cs.get_x(j), cs.get_y(j)) {
                min[0] = min[0].min(x);
                min[1] = min[1].min(y);
                max[0] = max[0].max(x);
                max[1] = max[1].max(y);
            }
        }

        let candidates = spatial::query_envelope(&tree, min, max);
        for j in candidates {
            if j <= i {
                continue;
            }
            let touches = if rook {
                // Rook: shared edge (not just point)
                polygons[i]
                    .intersection(&polygons[j])
                    .map(|inter| {
                        inter.length().unwrap_or(0.0) > 0.0
                    })
                    .unwrap_or(false)
            } else {
                // Queen: any shared boundary
                polygons[i].touches(&polygons[j]).unwrap_or(false)
                    || polygons[i].intersects(&polygons[j]).unwrap_or(false)
            };
            if touches {
                adjacency[i].push(j);
                adjacency[j].push(i);
            }
        }
    }

    adjacency
}

/// Compute connected component labels from an adjacency list.
pub fn component_labels_from_adjacency(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    let mut graph = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    for (i, neighbors) in adjacency.iter().enumerate() {
        for &j in neighbors {
            if j > i {
                graph.add_edge(nodes[i], nodes[j], ());
            }
        }
    }

    let mut labels = vec![0usize; n];
    let mut visited = vec![false; n];
    let mut label = 0;

    for start in 0..n {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;
            labels[node] = label;
            for neighbor in graph.neighbors(nodes[node]) {
                let nidx = neighbor.index();
                if !visited[nidx] {
                    stack.push(nidx);
                }
            }
        }
        label += 1;
    }

    labels
}

// ─── Internal helpers ───────────────────────────────────────────────────────

fn union_all_lines(geometries: &[GGeometry]) -> Option<GGeometry> {
    if geometries.is_empty() {
        return None;
    }
    let mut result = Clone::clone(&geometries[0]);
    for geom in &geometries[1..] {
        result = result.union(geom).ok()?;
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isoareal_quotient() {
        // Circle: area = π, perimeter = 2π → ratio = 1
        let ratio = isoareal_quotient(PI, 2.0 * PI);
        assert!((ratio - 1.0).abs() < 1e-10);

        // Square: area = 1, perimeter = 4 → ratio = 2π√(1/π) / 4 = √π/2 ≈ 0.886
        let ratio = isoareal_quotient(1.0, 4.0);
        let expected = PI.sqrt() / 2.0;
        assert!((ratio - expected).abs() < 1e-10);
    }

    #[test]
    fn test_isoperimetric_quotient() {
        // Circle: area = π, perimeter = 2π → ratio = 4π²/(4π²) = 1
        let ratio = isoperimetric_quotient(PI, 2.0 * PI);
        assert!((ratio - 1.0).abs() < 1e-10);

        // Square: area = 1, perimeter = 4 → ratio = 4π/16 = π/4 ≈ 0.785
        let ratio = isoperimetric_quotient(1.0, 4.0);
        assert!((ratio - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_kde() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (grid, pdf) = gaussian_kde(&data, 100);
        assert_eq!(grid.len(), 100);
        assert_eq!(pdf.len(), 100);
        // PDF should be non-negative
        assert!(pdf.iter().all(|&v| v >= 0.0));
        // PDF should peak near the center of the data
        let peak_idx = pdf
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let peak_x = grid[peak_idx];
        assert!((peak_x - 3.0).abs() < 2.0);
    }

    #[test]
    fn test_find_peaks_simple() {
        let signal = vec![0.0, 1.0, 0.5, 2.0, 0.0, 1.5, 0.0];
        let peaks = find_peaks(&signal, None, None);
        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
        assert!(peaks.contains(&5));
    }

    #[test]
    fn test_component_labels() {
        // 4 nodes: 0-1 connected, 2-3 connected
        let adjacency = vec![
            vec![1],    // 0 → 1
            vec![0],    // 1 → 0
            vec![3],    // 2 → 3
            vec![2],    // 3 → 2
        ];
        let labels = component_labels_from_adjacency(&adjacency);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_compute_enclosed_touching() {
        // 5 polygons: 0,1,2 form a chain, 3 adjacent to 1, 4 isolated
        // If 0 and 2 are artifacts:
        //   - 1 is enclosed (all neighbors 0,2 are artifacts)
        //   - 3 is touching (neighbor 1 is NOT artifact, so not enclosed, but
        //     it depends on adjacency)
        let adjacency = vec![
            vec![1],       // 0 → 1
            vec![0, 2, 3], // 1 → 0, 2, 3
            vec![1],       // 2 → 1
            vec![1],       // 3 → 1
            vec![],        // 4 (isolate)
        ];
        let is_artifact = vec![true, false, true, false, false];
        let is_isolate = vec![false, false, false, false, true];

        let (enclosed, touching) =
            compute_enclosed_touching(&adjacency, &is_artifact, &is_isolate);

        // Polygon 1: neighbors are [0, 2, 3]. Artifact flags: [true, true, false]
        // Not all neighbors are artifacts (3 is not), so enclosed=false
        // At least one neighbor is artifact (0 and 2), so touching=true
        assert!(!enclosed[1]);
        assert!(touching[1]);

        // Polygon 3: neighbors are [1]. is_artifact[1]=false
        // No artifact neighbors, so touching=false
        assert!(!enclosed[3]);
        assert!(!touching[3]);

        // Polygon 4: isolate → both false
        assert!(!enclosed[4]);
        assert!(!touching[4]);

        // Now make polygon 3 an artifact too:
        let is_artifact2 = vec![true, false, true, true, false];
        let (enclosed2, touching2) =
            compute_enclosed_touching(&adjacency, &is_artifact2, &is_isolate);

        // Polygon 1: neighbors [0, 2, 3] all are artifacts → enclosed=true
        assert!(enclosed2[1]);
        assert!(touching2[1]);
    }
}
