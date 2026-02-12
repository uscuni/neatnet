//! Face artifact detection and classification.
//!
//! Ports Python `neatnet.artifacts`: `FaceArtifacts`, `get_artifacts`,
//! and all the artifact processing functions (n1_g1, nx_gx, etc.).

use std::f64::consts::PI;

use geos::{Geom, Geometry as GGeometry};
use petgraph::graph::UnGraph;

use crate::spatial;

/// Shape metric: isoareal quotient = 4πA/P².
/// Ranges from 0 (elongated) to 1 (circular).
pub fn isoareal_quotient(area: f64, perimeter: f64) -> f64 {
    if perimeter <= 0.0 {
        return 0.0;
    }
    4.0 * PI * area / (perimeter * perimeter)
}

/// Shape metric: isoperimetric quotient = P / (2π√(A/π)).
/// Ratio of perimeter to circumference of equal-area circle.
pub fn isoperimetric_quotient(area: f64, perimeter: f64) -> f64 {
    if area <= 0.0 {
        return 0.0;
    }
    perimeter / (2.0 * PI * (area / PI).sqrt())
}

/// Minimum bounding circle ratio (approximation).
/// Uses the convex hull area as a proxy for the minimum bounding circle area.
pub fn minimum_bounding_circle_ratio(geom: &GGeometry) -> f64 {
    let area = geom.area().unwrap_or(0.0);
    if area <= 0.0 {
        return 0.0;
    }
    // The minimum bounding circle ratio is area / MBC_area
    // Approximate MBC via convex hull
    let hull_area = geom
        .convex_hull()
        .and_then(|h| h.area())
        .unwrap_or(area);
    if hull_area <= 0.0 {
        return 0.0;
    }
    area / hull_area
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
        let cs = match env.get_coord_seq() {
            Ok(c) => c,
            Err(_) => continue,
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

        // Square: area = 1, perimeter = 4 → ratio = π/4 ≈ 0.785
        let ratio = isoareal_quotient(1.0, 4.0);
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
}
