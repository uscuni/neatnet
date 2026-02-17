//! Face artifact detection and classification.
//!
//! Ports Python `neatnet.artifacts`: `FaceArtifacts`, `get_artifacts`,
//! and all the artifact processing functions (n1_g1, nx_gx, etc.).

use std::f64::consts::PI;

use geo::{Area, BoundingRect, Euclidean, Intersects, Length, Relate};
use geo_types::{LineString, Polygon};
use petgraph::graph::UnGraph;

use crate::ops;
use crate::spatial;

/// Shape metric: isoareal quotient (Altman's PA_3).
pub fn isoareal_quotient(area: f64, perimeter: f64) -> f64 {
    if area <= 0.0 || perimeter <= 0.0 {
        return 0.0;
    }
    2.0 * PI * (area / PI).sqrt() / perimeter
}

/// Shape metric: isoperimetric quotient (Altman's PA_1).
pub fn isoperimetric_quotient(area: f64, perimeter: f64) -> f64 {
    if perimeter <= 0.0 {
        return 0.0;
    }
    4.0 * PI * area / (perimeter * perimeter)
}

/// Minimum bounding circle ratio (Reock compactness).
pub fn minimum_bounding_circle_ratio(geom: &Polygon<f64>) -> f64 {
    let area = geom.unsigned_area();
    if area <= 0.0 {
        return 0.0;
    }

    // Get convex hull vertices
    use geo::ConvexHull;
    let hull = geom.convex_hull();
    let ring = hull.exterior();
    let coords = &ring.0;
    if coords.len() < 4 {
        return 0.0;
    }

    // Extract hull vertex coordinates (skip last = first for closed ring)
    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(coords.len() - 1);
    for c in &coords[..coords.len().saturating_sub(1)] {
        pts.push([c.x, c.y]);
    }

    let radius = minimum_enclosing_circle_radius(&pts);
    let mbc_area = PI * radius * radius;
    if mbc_area <= 0.0 {
        return 0.0;
    }

    area / mbc_area
}

/// Compute the radius of the minimum enclosing circle for a set of 2D points.
fn minimum_enclosing_circle_radius(points: &[[f64; 2]]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    if points.len() == 1 {
        return 0.0;
    }

    let mut indices: Vec<usize> = (0..points.len()).collect();
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
            continue;
        }

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
                continue;
            }

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
                    continue;
                }

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

fn circumcircle(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> Option<(f64, f64, f64)> {
    let ax = a[0]; let ay = a[1];
    let bx = b[0]; let by = b[1];
    let cx = c[0]; let cy = c[1];

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-14 {
        return None;
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)) / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)) / d;

    let dx = ax - ux;
    let dy = ay - uy;
    Some((ux, uy, dx * dx + dy * dy))
}

/// Gaussian Kernel Density Estimation with Silverman bandwidth.
pub fn gaussian_kde(data: &[f64], n_points: usize) -> (Vec<f64>, Vec<f64>) {
    let n = data.len() as f64;
    if data.is_empty() {
        return (vec![], vec![]);
    }

    let mean: f64 = data.iter().sum::<f64>() / n;
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

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
            let sum: f64 = data.iter()
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

/// Find peaks in a 1-D signal.
pub fn find_peaks(signal: &[f64], min_height: Option<f64>, min_prominence: Option<f64>) -> Vec<usize> {
    let n = signal.len();
    if n < 3 {
        return vec![];
    }

    let mut peaks: Vec<usize> = Vec::new();
    for i in 1..n - 1 {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            peaks.push(i);
        }
    }

    if let Some(height) = min_height {
        peaks.retain(|&i| signal[i] >= height);
    }

    if let Some(prominence) = min_prominence {
        peaks.retain(|&peak| compute_prominence(signal, peak) >= prominence);
    }

    peaks
}

fn compute_prominence(signal: &[f64], peak: usize) -> f64 {
    let peak_val = signal[peak];
    let mut left_min = peak_val;
    for i in (0..peak).rev() {
        left_min = left_min.min(signal[i]);
        if signal[i] > peak_val { break; }
    }
    let mut right_min = peak_val;
    for i in (peak + 1)..signal.len() {
        right_min = right_min.min(signal[i]);
        if signal[i] > peak_val { break; }
    }
    peak_val - left_min.max(right_min)
}

/// Full artifact detection with iterative expansion.
pub fn get_artifacts(
    geometries: &[LineString<f64>],
    threshold: Option<f64>,
    threshold_fallback: f64,
    exclusion_mask: Option<&[Polygon<f64>]>,
    area_threshold_blocks: f64,
    isoareal_threshold_blocks: f64,
    area_threshold_circles: f64,
    isoareal_threshold_circles_enclosed: f64,
    isoperimetric_threshold_circles_touching: f64,
) -> Option<(Vec<Polygon<f64>>, Vec<f64>, f64)> {
    // 1. Polygonize
    let poly_geoms = ops::polygonize(geometries);

    if poly_geoms.is_empty() {
        return None;
    }

    // 2. Compute FAI values
    let mut fai_values = Vec::with_capacity(poly_geoms.len());
    for poly in &poly_geoms {
        let area = poly.unsigned_area();
        let mbc_ratio = minimum_bounding_circle_ratio(poly);
        let fai = (mbc_ratio * area).ln();
        fai_values.push(fai);
    }

    // 3. Determine FAI threshold
    let final_threshold = if let Some(t) = threshold {
        t
    } else {
        match find_fai_threshold(&fai_values) {
            Some(t) => t,
            None => threshold_fallback,
        }
    };

    // 4. Initialize is_artifact flags
    let mut is_artifact: Vec<bool> = fai_values.iter().map(|&fai| fai < final_threshold).collect();

    // 5. Pre-compute shape metrics
    let areas: Vec<f64> = poly_geoms.iter().map(|g| g.unsigned_area()).collect();
    let perimeters: Vec<f64> = poly_geoms.iter().map(|g| Euclidean.length(g.exterior())).collect();
    let isoareal: Vec<f64> = areas.iter().zip(perimeters.iter())
        .map(|(&a, &p)| isoareal_quotient(a, p)).collect();
    let isoperimetric: Vec<f64> = areas.iter().zip(perimeters.iter())
        .map(|(&a, &p)| isoperimetric_quotient(a, p)).collect();

    // 6. Build rook contiguity graph
    let adjacency = build_contiguity_graph(&poly_geoms, true);
    let is_isolate: Vec<bool> = adjacency.iter().map(|adj| adj.is_empty()).collect();

    // 7. Iterative expansion
    loop {
        let artifact_count_before: usize = is_artifact.iter().filter(|&&a| a).count();

        let (enclosed, touching) =
            compute_enclosed_touching(&adjacency, &is_artifact, &is_isolate);

        for i in 0..poly_geoms.len() {
            if is_artifact[i] { continue; }
            if (enclosed[i] || touching[i])
                && areas[i] < area_threshold_blocks
                && isoareal[i] < isoareal_threshold_blocks
            {
                is_artifact[i] = true;
            }
        }

        for i in 0..poly_geoms.len() {
            if is_artifact[i] { continue; }
            if enclosed[i]
                && areas[i] < area_threshold_circles
                && isoareal[i] > isoareal_threshold_circles_enclosed
            {
                is_artifact[i] = true;
            }
        }

        for i in 0..poly_geoms.len() {
            if is_artifact[i] { continue; }
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

    // 8. Apply exclusion mask
    if let Some(mask) = exclusion_mask {
        for i in 0..poly_geoms.len() {
            if !is_artifact[i] { continue; }
            for mask_geom in mask {
                if poly_geoms[i].intersects(mask_geom) {
                    is_artifact[i] = false;
                    break;
                }
            }
        }
    }

    // 9. Collect artifact polygons
    let mut artifact_geoms = Vec::new();
    let mut artifact_fais = Vec::new();
    for (i, geom) in poly_geoms.into_iter().enumerate() {
        if is_artifact[i] {
            artifact_fais.push(fai_values[i]);
            artifact_geoms.push(geom);
        }
    }

    Some((artifact_geoms, artifact_fais, final_threshold))
}

fn compute_enclosed_touching(
    adjacency: &[Vec<usize>],
    is_artifact: &[bool],
    is_isolate: &[bool],
) -> (Vec<bool>, Vec<bool>) {
    let n = adjacency.len();
    let mut enclosed = vec![false; n];
    let mut touching = vec![false; n];

    for i in 0..n {
        if is_isolate[i] || is_artifact[i] { continue; }
        let neighbors = &adjacency[i];
        if neighbors.is_empty() { continue; }
        enclosed[i] = neighbors.iter().all(|&j| is_artifact[j]);
        touching[i] = neighbors.iter().any(|&j| is_artifact[j]);
    }

    (enclosed, touching)
}

fn find_fai_threshold(fai_values: &[f64]) -> Option<f64> {
    let (grid, pdf) = gaussian_kde(fai_values, 1000);
    let peaks = find_peaks(&pdf, Some(0.008), Some(0.00075));
    if peaks.len() < 2 { return None; }

    let inverted: Vec<f64> = pdf.iter().map(|&v| -v + 1.0).collect();
    let valleys = find_peaks(&inverted, None, Some(0.00075));
    if valleys.is_empty() { return None; }

    let highest_peak_idx = peaks.iter()
        .max_by(|&&a, &&b| pdf[a].partial_cmp(&pdf[b]).unwrap())?;

    for &valley in &valleys {
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
pub fn build_contiguity_graph(polygons: &[Polygon<f64>], rook: bool) -> Vec<Vec<usize>> {
    let n = polygons.len();
    let tree = spatial::build_rtree_polys(polygons);
    let mut adjacency: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        let rect = match polygons[i].bounding_rect() {
            Some(r) => r,
            None => continue,
        };
        let min = [rect.min().x, rect.min().y];
        let max = [rect.max().x, rect.max().y];

        let candidates = spatial::query_envelope(&tree, min, max);
        for j in candidates {
            if j <= i { continue; }
            let touches = if rook {
                // Rook: shared edge (intersection has length > 0)
                // For rook, we need shared edges. Use relate to check.
                let de9im = polygons[i].relate(&polygons[j]);
                // Rook adjacency = shared boundary is 1-dimensional (line, not just point)
                // DE-9IM: check if boundary-boundary intersection is at least 1D
                {
                    use geo::coordinate_position::CoordPos;
                    de9im.get(CoordPos::OnBoundary, CoordPos::OnBoundary) == geo::dimensions::Dimensions::OneDimensional
                        || de9im.get(CoordPos::OnBoundary, CoordPos::OnBoundary) == geo::dimensions::Dimensions::TwoDimensional
                }
            } else {
                polygons[i].relate(&polygons[j]).is_touches()
                    || polygons[i].intersects(&polygons[j])
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
        if visited[start] { continue; }
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited[node] { continue; }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isoareal_quotient() {
        let ratio = isoareal_quotient(PI, 2.0 * PI);
        assert!((ratio - 1.0).abs() < 1e-10);

        let ratio = isoareal_quotient(1.0, 4.0);
        let expected = PI.sqrt() / 2.0;
        assert!((ratio - expected).abs() < 1e-10);
    }

    #[test]
    fn test_isoperimetric_quotient() {
        let ratio = isoperimetric_quotient(PI, 2.0 * PI);
        assert!((ratio - 1.0).abs() < 1e-10);

        let ratio = isoperimetric_quotient(1.0, 4.0);
        assert!((ratio - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_kde() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (grid, pdf) = gaussian_kde(&data, 100);
        assert_eq!(grid.len(), 100);
        assert_eq!(pdf.len(), 100);
        assert!(pdf.iter().all(|&v| v >= 0.0));
        let peak_idx = pdf.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap().0;
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
        let adjacency = vec![vec![1], vec![0], vec![3], vec![2]];
        let labels = component_labels_from_adjacency(&adjacency);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_compute_enclosed_touching() {
        let adjacency = vec![
            vec![1],
            vec![0, 2, 3],
            vec![1],
            vec![1],
            vec![],
        ];
        let is_artifact = vec![true, false, true, false, false];
        let is_isolate = vec![false, false, false, false, true];

        let (enclosed, touching) =
            compute_enclosed_touching(&adjacency, &is_artifact, &is_isolate);

        assert!(!enclosed[1]);
        assert!(touching[1]);
        assert!(!enclosed[3]);
        assert!(!touching[3]);
        assert!(!enclosed[4]);
        assert!(!touching[4]);

        let is_artifact2 = vec![true, false, true, true, false];
        let (enclosed2, touching2) =
            compute_enclosed_touching(&adjacency, &is_artifact2, &is_isolate);
        assert!(enclosed2[1]);
        assert!(touching2[1]);
    }
}
