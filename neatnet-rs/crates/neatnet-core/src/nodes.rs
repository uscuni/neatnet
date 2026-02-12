//! Node operations: topology fixing, node consolidation, component labeling.
//!
//! Ports Python `neatnet.nodes`: `fix_topology`, `consolidate_nodes`,
//! `get_components`, `remove_interstitial_nodes`, `induce_nodes`.

use std::collections::HashMap;

use geos::{Geom, Geometry as GGeometry};
use petgraph::graph::UnGraph;
use petgraph::algo::connected_components;

use crate::types::EdgeStatus;

/// Rounded coordinate key for endpoint hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct NodeKey {
    x: i64,
    y: i64,
}

fn node_key(x: f64, y: f64) -> NodeKey {
    NodeKey {
        x: (x * 1e8).round() as i64,
        y: (y * 1e8).round() as i64,
    }
}

/// Extract unique network nodes (endpoints) from edge geometries.
///
/// Returns `(node_points, degrees)` where degree counts how many edge
/// endpoints touch each unique node coordinate.
pub fn nodes_from_edges(
    geometries: &[GGeometry],
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut node_counts: HashMap<NodeKey, (f64, f64, usize)> = HashMap::new();

    for geom in geometries {
        let Ok(cs) = geom.get_coord_seq() else { continue };
        let Ok(n) = cs.size() else { continue };
        if n < 2 {
            continue;
        }
        // Start point
        if let (Ok(x), Ok(y)) = (cs.get_x(0), cs.get_y(0)) {
            let key = node_key(x, y);
            let entry = node_counts.entry(key).or_insert((x, y, 0));
            entry.2 += 1;
        }
        // End point
        if let (Ok(x), Ok(y)) = (cs.get_x(n - 1), cs.get_y(n - 1)) {
            let key = node_key(x, y);
            let entry = node_counts.entry(key).or_insert((x, y, 0));
            entry.2 += 1;
        }
    }

    let mut coords = Vec::with_capacity(node_counts.len());
    let mut degrees = Vec::with_capacity(node_counts.len());
    for (_, (x, y, deg)) in &node_counts {
        coords.push([*x, *y]);
        degrees.push(*deg);
    }
    (coords, degrees)
}

/// Determine component labels for chains of edges linked by degree-2 nodes.
///
/// Edges sharing a degree-2 node (that is not closed/looped) are assigned
/// the same component label. These are later merged by `line_merge`.
///
/// Mirrors Python `get_components()`.
pub fn get_components(geometries: &[GGeometry]) -> Vec<usize> {
    let n = geometries.len();
    if n == 0 {
        return vec![];
    }

    // 1. Extract all unique node coordinates
    let (node_coords, _) = nodes_from_edges(geometries);

    // 2. Build node → [edge indices] map from boundary intersections
    let mut node_to_edges: HashMap<NodeKey, Vec<usize>> = HashMap::new();
    for (edge_idx, geom) in geometries.iter().enumerate() {
        let Ok(cs) = geom.get_coord_seq() else { continue };
        let Ok(npts) = cs.size() else { continue };
        if npts < 2 {
            continue;
        }
        if let (Ok(x), Ok(y)) = (cs.get_x(0), cs.get_y(0)) {
            node_to_edges
                .entry(node_key(x, y))
                .or_default()
                .push(edge_idx);
        }
        if let (Ok(x), Ok(y)) = (cs.get_x(npts - 1), cs.get_y(npts - 1)) {
            node_to_edges
                .entry(node_key(x, y))
                .or_default()
                .push(edge_idx);
        }
    }

    // 3. Identify closed (loop) edges
    let is_closed: Vec<bool> = geometries
        .iter()
        .map(|g| g.is_ring().unwrap_or(false))
        .collect();

    // 4. Find degree-2 nodes that connect exactly 2 non-loop edges
    //    Build a graph where edges that share a degree-2 node are connected
    let mut graph = UnGraph::<(), ()>::new_undirected();
    let graph_nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    for (_key, edge_indices) in &node_to_edges {
        if edge_indices.len() != 2 {
            continue;
        }
        let e0 = edge_indices[0];
        let e1 = edge_indices[1];
        // Skip if either edge is a loop
        if is_closed[e0] || is_closed[e1] {
            continue;
        }
        graph.add_edge(graph_nodes[e0], graph_nodes[e1], ());
    }

    // 5. Extract connected components as labels
    let mut labels = vec![0usize; n];
    let components = connected_components(&graph);
    let _ = components; // the function returns count, we need per-node labels

    // Use petgraph's built-in DFS-based component assignment
    let mut component_map: HashMap<usize, usize> = HashMap::new();
    let mut visited = vec![false; n];
    let mut label = 0;

    for start in 0..n {
        if visited[start] {
            continue;
        }
        // BFS/DFS through the component graph
        let mut stack = vec![start];
        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }
            visited[node] = true;
            labels[node] = label;
            component_map.insert(node, label);

            // Visit neighbors in the merge graph
            for neighbor in graph.neighbors(graph_nodes[node]) {
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

/// Remove interstitial (degree-2) nodes by merging edge chains.
///
/// Returns the merged geometries and aggregated statuses.
pub fn remove_interstitial_nodes<'a>(
    geometries: &[GGeometry],
    statuses: &[EdgeStatus],
) -> (Vec<GGeometry>, Vec<EdgeStatus>) {
    if geometries.len() < 2 {
        return (geometries.to_vec(), statuses.to_vec());
    }

    let labels = get_components(geometries);

    // Group geometries by component label
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        groups.entry(label).or_default().push(idx);
    }

    let mut result_geoms = Vec::new();
    let mut result_statuses = Vec::new();

    for (_label, indices) in &groups {
        if indices.len() == 1 {
            // Single edge in component, keep as-is
            result_geoms.push(Clone::clone(&geometries[indices[0]]));
            result_statuses.push(statuses[indices[0]]);
        } else {
            // Multiple edges: collect and line_merge
            let group_statuses: Vec<EdgeStatus> =
                indices.iter().map(|&i| statuses[i]).collect();
            let merged_status = EdgeStatus::aggregate(&group_statuses);

            // Build a GeometryCollection and line_merge
            match merge_linestrings(indices.iter().map(|&i| &geometries[i])) {
                Some(merged) => {
                    result_geoms.push(merged);
                    result_statuses.push(merged_status);
                }
                None => {
                    // Fallback: keep individual geometries
                    for &idx in indices {
                        result_geoms.push(Clone::clone(&geometries[idx]));
                        result_statuses.push(statuses[idx]);
                    }
                }
            }
        }
    }

    (result_geoms, result_statuses)
}

/// Merge a set of LineString geometries using GEOS line_merge.
fn merge_linestrings<'b, I>(geoms: I) -> Option<GGeometry>
where
    I: Iterator<Item = &'b GGeometry>,
{
    let parts: Vec<_> = geoms.collect();
    if parts.is_empty() {
        return None;
    }
    if parts.len() == 1 {
        return Some(Clone::clone(parts[0]));
    }

    // Build WKT for a GeometryCollection
    let mut wkt_parts = Vec::new();
    for part in &parts {
        if let Ok(w) = part.to_wkt() {
            wkt_parts.push(w);
        }
    }
    let collection_wkt = format!(
        "GEOMETRYCOLLECTION ({})",
        wkt_parts.join(", ")
    );

    let collection = GGeometry::new_from_wkt(&collection_wkt).ok()?;
    collection.line_merge().ok()
}

/// Fix street network topology.
///
/// 1. Remove duplicate geometries (normalized).
/// 2. Induce missing nodes at intersections.
/// 3. Remove interstitial (degree-2) nodes.
///
/// Mirrors Python `fix_topology()`.
pub fn fix_topology<'a>(
    geometries: &[GGeometry],
    statuses: &[EdgeStatus],
    _eps: f64,
) -> (Vec<GGeometry>, Vec<EdgeStatus>) {
    // Step 1: Remove duplicates (by normalized WKT)
    let mut seen = std::collections::HashSet::new();
    let mut deduped_geoms = Vec::new();
    let mut deduped_statuses = Vec::new();

    for (geom, &status) in geometries.iter().zip(statuses.iter()) {
        let mut normalized = Clone::clone(geom);
        if normalized.normalize().is_ok() {
            if let Ok(wkt) = normalized.to_wkt() {
                if seen.insert(wkt) {
                    deduped_geoms.push(Clone::clone(geom));
                    deduped_statuses.push(status);
                }
            }
        }
    }

    // Step 2: Induce nodes at intersections
    // (Full implementation requires spatial index queries – stubbed for now,
    //  will be completed in Phase 3)

    // Step 3: Remove interstitial nodes
    remove_interstitial_nodes(&deduped_geoms, &deduped_statuses)
}

/// Consolidate nearby nodes using hierarchical clustering.
///
/// Replaces clusters of nodes within `tolerance` distance with a single
/// weighted centroid node, generating "spider" geometry to maintain
/// connectivity.
///
/// Mirrors Python `consolidate_nodes()`.
pub fn consolidate_nodes<'a>(
    geometries: &[GGeometry],
    statuses: &[EdgeStatus],
    tolerance: f64,
    _preserve_ends: bool,
) -> (Vec<GGeometry>, Vec<EdgeStatus>) {
    let (node_coords, degrees) = nodes_from_edges(geometries);

    if node_coords.len() < 2 {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // Use kodama hierarchical clustering with average linkage
    let n = node_coords.len();

    // Build condensed distance matrix for kodama
    let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = node_coords[i][0] - node_coords[j][0];
            let dy = node_coords[i][1] - node_coords[j][1];
            condensed.push((dx * dx + dy * dy).sqrt() as f32);
        }
    }

    let dendrogram = kodama::linkage(&mut condensed, n, kodama::Method::Average);

    // Cut the dendrogram at `tolerance` distance to get cluster labels
    let cluster_labels = fcluster(&dendrogram, tolerance as f32, n);


    // Find clusters with more than one node
    let mut cluster_sizes: HashMap<usize, usize> = HashMap::new();
    for &label in &cluster_labels {
        *cluster_sizes.entry(label).or_default() += 1;
    }

    let has_clusters = cluster_sizes.values().any(|&size| size > 1);
    if !has_clusters {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // TODO: Full spider geometry generation.
    // For now, return geometries unchanged (will be completed in Phase 3).
    (geometries.to_vec(), statuses.to_vec())
}

/// Cut a kodama dendrogram at a given distance threshold, returning cluster labels.
///
/// Equivalent to `scipy.cluster.hierarchy.fcluster(..., criterion='distance')`.
fn fcluster(dendrogram: &kodama::Dendrogram<f32>, threshold: f32, n: usize) -> Vec<usize> {
    // Each original observation starts as its own cluster
    let mut labels = vec![0usize; n];
    let mut next_label = 0;

    // Union-Find approach: walk the dendrogram and merge clusters
    // whose linkage distance is <= threshold
    let mut parent: Vec<usize> = (0..2 * n).collect();

    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for (step_idx, step) in dendrogram.steps().iter().enumerate() {
        let new_cluster = n + step_idx;
        if step.dissimilarity <= threshold {
            // Merge: both children belong to the same cluster
            let a = find(&mut parent, step.cluster1);
            let b = find(&mut parent, step.cluster2);
            parent[a] = new_cluster;
            parent[b] = new_cluster;
            parent[new_cluster] = new_cluster;
        } else {
            // Don't merge
            parent[new_cluster] = new_cluster;
        }
    }

    // Assign labels: each root gets a unique label
    let mut root_to_label: HashMap<usize, usize> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        let label = *root_to_label.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels[i] = label;
    }

    labels
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
    fn test_nodes_from_edges() {
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let (coords, degrees) = nodes_from_edges(&[g1, g2]);
        // 3 unique nodes: (0,0) deg 1, (1,0) deg 2, (2,0) deg 1
        assert_eq!(coords.len(), 3);
        assert_eq!(degrees.len(), 3);
    }

    #[test]
    fn test_get_components_chain() {
        // Three edges in a chain: 0-1-2-3
        // Nodes at 1 and 2 are degree-2 → all three should share a component
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let g3 = make_line(&[[2.0, 0.0], [3.0, 0.0]]);
        let labels = get_components(&[g1, g2, g3]);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn test_get_components_branch() {
        // T-junction: node at (1,0) has degree 3 → separate components
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let g3 = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let labels = get_components(&[g1, g2, g3]);
        // g1 and g2 share degree-2 node at (1,0)? No – (1,0) has degree 3
        // so each edge gets its own component
        assert!(labels[0] != labels[2] || labels[1] != labels[2]);
    }

    #[test]
    fn test_fcluster_basic() {
        // 4 points: 0=(0,0), 1=(1,0), 2=(10,0), 3=(11,0)
        // With threshold 2, should get 2 clusters: {0,1} and {2,3}
        let mut condensed = vec![
            1.0f32,  // d(0,1)
            10.0,    // d(0,2)
            11.0,    // d(0,3)
            9.0,     // d(1,2)
            10.0,    // d(1,3)
            1.0,     // d(2,3)
        ];
        let dendro = kodama::linkage(&mut condensed, 4, kodama::Method::Average);
        let labels = fcluster(&dendro, 2.0, 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }
}
