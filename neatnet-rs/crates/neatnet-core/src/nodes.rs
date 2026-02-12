//! Node operations: topology fixing, node consolidation, component labeling.
//!
//! Ports Python `neatnet.nodes`: `fix_topology`, `consolidate_nodes`,
//! `get_components`, `remove_interstitial_nodes`, `induce_nodes`.

use std::collections::HashMap;

use geos::{Geom, Geometry as GGeometry, GeometryTypes};
use petgraph::algo::connected_components;
use petgraph::graph::UnGraph;
use rstar::primitives::GeomWithData;
use rstar::{RTree, AABB};

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
    let (induced_geoms, induced_statuses) = induce_nodes(&deduped_geoms, &deduped_statuses, _eps);

    // Step 3: Remove interstitial nodes
    remove_interstitial_nodes(&induced_geoms, &induced_statuses)
}

/// Add missing nodes where line endpoints intersect other edges.
///
/// If a line endpoint touches (within `eps`) another edge that doesn't
/// have a vertex at that point, we split the other edge to create a
/// proper network node.
///
/// Mirrors Python `induce_nodes()`.
pub fn induce_nodes(
    geometries: &[GGeometry],
    statuses: &[EdgeStatus],
    eps: f64,
) -> (Vec<GGeometry>, Vec<EdgeStatus>) {
    if geometries.is_empty() {
        return (vec![], vec![]);
    }

    // 1. Identify degree mismatches
    let mismatch_points = identify_degree_mismatch(geometries, eps);

    // 2. Identify loop contact points
    let (loop_non_loop_pts, loop_loop_pts) = makes_loop_contact(geometries, eps);

    // 3. Collect all split points (deduplicated)
    let mut all_split_points: Vec<GGeometry> = Vec::new();
    let mut seen_keys = std::collections::HashSet::new();

    for pt in mismatch_points
        .iter()
        .chain(loop_non_loop_pts.iter())
        .chain(loop_loop_pts.iter())
    {
        if let Ok(cs) = pt.get_coord_seq() {
            if let (Ok(x), Ok(y)) = (cs.get_x(0), cs.get_y(0)) {
                let key = node_key(x, y);
                if seen_keys.insert(key) {
                    all_split_points.push(Clone::clone(pt));
                }
            }
        }
    }

    if all_split_points.is_empty() {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // 4. Split edges at all identified points
    split_edges_at_points(geometries, statuses, &all_split_points, eps)
}

/// Identify nodes where observed degree != expected degree.
///
/// Observed degree: how many edge endpoints are at that coordinate.
/// Expected degree: how many edges actually pass through that point (dwithin eps).
fn identify_degree_mismatch(geometries: &[GGeometry], eps: f64) -> Vec<GGeometry> {
    let (node_coords, degrees) = nodes_from_edges(geometries);
    let geom_tree = crate::spatial::build_rtree(geometries);

    let mut result = Vec::new();
    for (i, coord) in node_coords.iter().enumerate() {
        let candidates = crate::spatial::query_envelope(
            &geom_tree,
            [coord[0] - eps, coord[1] - eps],
            [coord[0] + eps, coord[1] + eps],
        );

        // Count edges actually within eps of this node
        let pt_wkt = format!("POINT ({} {})", coord[0], coord[1]);
        let pt = match GGeometry::new_from_wkt(&pt_wkt) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let mut expected = 0usize;
        for &idx in &candidates {
            if let Ok(dist) = geometries[idx].distance(&pt) {
                if dist <= eps {
                    expected += 1;
                }
            }
        }

        if expected != degrees[i] {
            result.push(pt);
        }
    }
    result
}

/// Identify loop vertices that contact non-loop edges or other loops.
fn makes_loop_contact(
    geometries: &[GGeometry],
    eps: f64,
) -> (Vec<GGeometry>, Vec<GGeometry>) {
    let mut loop_indices = Vec::new();
    let mut non_loop_indices = Vec::new();

    for (i, geom) in geometries.iter().enumerate() {
        if geom.is_ring().unwrap_or(false) {
            loop_indices.push(i);
        } else {
            non_loop_indices.push(i);
        }
    }

    if loop_indices.is_empty() {
        return (vec![], vec![]);
    }

    // Build R-tree of non-loop edges
    let non_loop_geoms: Vec<_> = non_loop_indices.iter().map(|&i| Clone::clone(&geometries[i])).collect();
    let non_loop_tree = crate::spatial::build_rtree(&non_loop_geoms);

    let loop_geoms: Vec<_> = loop_indices.iter().map(|&i| Clone::clone(&geometries[i])).collect();
    let loop_tree = crate::spatial::build_rtree(&loop_geoms);

    let mut non_loop_contact = Vec::new();
    let mut loop_contact = Vec::new();

    // Extract all vertices from loop geometries
    for &li in &loop_indices {
        let geom = &geometries[li];
        let Ok(cs) = geom.get_coord_seq() else { continue };
        let Ok(n) = cs.size() else { continue };

        for vi in 0..n {
            let Ok(x) = cs.get_x(vi) else { continue };
            let Ok(y) = cs.get_y(vi) else { continue };

            let pt_wkt = format!("POINT ({} {})", x, y);
            let pt = match GGeometry::new_from_wkt(&pt_wkt) {
                Ok(p) => p,
                Err(_) => continue,
            };

            // Check if this loop vertex touches a non-loop edge
            let nl_candidates = crate::spatial::query_envelope(
                &non_loop_tree,
                [x - eps, y - eps],
                [x + eps, y + eps],
            );
            for &idx in &nl_candidates {
                if let Ok(dist) = non_loop_geoms[idx].distance(&pt) {
                    if dist <= eps {
                        non_loop_contact.push(Clone::clone(&pt));
                        break;
                    }
                }
            }

            // Check if this loop vertex touches another loop
            let l_candidates = crate::spatial::query_envelope(
                &loop_tree,
                [x - eps, y - eps],
                [x + eps, y + eps],
            );
            let mut touch_count = 0;
            for &idx in &l_candidates {
                if let Ok(dist) = loop_geoms[idx].distance(&pt) {
                    if dist <= eps {
                        touch_count += 1;
                    }
                }
            }
            if touch_count > 1 {
                // This point is on 2+ loops
                loop_contact.push(Clone::clone(&pt));
            }
        }
    }

    (non_loop_contact, loop_contact)
}

/// Split edges at given split points.
///
/// For each split point, find edges within `eps` and split them.
fn split_edges_at_points(
    geometries: &[GGeometry],
    statuses: &[EdgeStatus],
    split_points: &[GGeometry],
    eps: f64,
) -> (Vec<GGeometry>, Vec<EdgeStatus>) {
    let mut result_geoms: Vec<GGeometry> = geometries.to_vec();
    let mut result_statuses: Vec<EdgeStatus> = statuses.to_vec();

    for split_pt in split_points {
        let split_cs = match split_pt.get_coord_seq() {
            Ok(cs) => cs,
            Err(_) => continue,
        };
        let (sx, sy) = match (split_cs.get_x(0), split_cs.get_y(0)) {
            (Ok(x), Ok(y)) => (x, y),
            _ => continue,
        };

        // Find edges near this split point
        let tree = crate::spatial::build_rtree(&result_geoms);
        let candidates = crate::spatial::query_envelope(
            &tree,
            [sx - eps * 10.0, sy - eps * 10.0],
            [sx + eps * 10.0, sy + eps * 10.0],
        );

        let mut to_remove = Vec::new();
        let mut to_add_geoms = Vec::new();
        let mut to_add_statuses = Vec::new();

        for &idx in &candidates {
            let geom = &result_geoms[idx];
            if let Ok(dist) = geom.distance(split_pt) {
                if dist > eps {
                    continue;
                }
            } else {
                continue;
            }

            // Try to split this edge
            let parts = snap_n_split(geom, split_pt, eps);
            if parts.len() > 1 {
                to_remove.push(idx);
                for part in parts {
                    to_add_geoms.push(part);
                    to_add_statuses.push(EdgeStatus::Changed);
                }
            }
        }

        // Apply removals and additions (in reverse order to preserve indices)
        if !to_remove.is_empty() {
            to_remove.sort_unstable();
            to_remove.dedup();
            for &idx in to_remove.iter().rev() {
                result_geoms.remove(idx);
                result_statuses.remove(idx);
            }
            result_geoms.extend(to_add_geoms);
            result_statuses.extend(to_add_statuses);
        }
    }

    (result_geoms, result_statuses)
}

/// Snap a point onto a line and split the line at that point.
///
/// Returns the split parts (1 element if split failed, 2+ on success).
fn snap_n_split(edge: &GGeometry, point: &GGeometry, eps: f64) -> Vec<GGeometry> {
    // Snap the edge to the point so the point becomes a vertex
    let snapped = match edge.snap(point, eps) {
        Ok(s) => s,
        Err(_) => return vec![Clone::clone(edge)],
    };

    let Ok(cs) = snapped.get_coord_seq() else {
        return vec![Clone::clone(edge)];
    };
    let Ok(n) = cs.size() else {
        return vec![Clone::clone(edge)];
    };
    if n < 2 {
        return vec![Clone::clone(edge)];
    }

    // Find the index of the snapped point in the coordinate sequence
    let pt_cs = match point.get_coord_seq() {
        Ok(c) => c,
        Err(_) => return vec![Clone::clone(edge)],
    };
    let (px, py) = match (pt_cs.get_x(0), pt_cs.get_y(0)) {
        (Ok(x), Ok(y)) => (x, y),
        _ => return vec![Clone::clone(edge)],
    };

    let mut split_idx = None;
    for i in 0..n {
        if let (Ok(x), Ok(y)) = (cs.get_x(i), cs.get_y(i)) {
            let dx = x - px;
            let dy = y - py;
            if (dx * dx + dy * dy).sqrt() < eps * 2.0 {
                // Don't split at endpoints
                if i > 0 && i < n - 1 {
                    split_idx = Some(i);
                    break;
                }
            }
        }
    }

    let Some(idx) = split_idx else {
        return vec![Clone::clone(edge)];
    };

    // Build two sub-lines: [0..idx] and [idx..n-1]
    let mut coords1 = Vec::new();
    for i in 0..=idx {
        if let (Ok(x), Ok(y)) = (cs.get_x(i), cs.get_y(i)) {
            coords1.push(format!("{} {}", x, y));
        }
    }
    let mut coords2 = Vec::new();
    for i in idx..n {
        if let (Ok(x), Ok(y)) = (cs.get_x(i), cs.get_y(i)) {
            coords2.push(format!("{} {}", x, y));
        }
    }

    let mut parts = Vec::new();
    if coords1.len() >= 2 {
        let wkt = format!("LINESTRING ({})", coords1.join(", "));
        if let Ok(g) = GGeometry::new_from_wkt(&wkt) {
            if !g.is_empty().unwrap_or(true) {
                parts.push(g);
            }
        }
    }
    if coords2.len() >= 2 {
        let wkt = format!("LINESTRING ({})", coords2.join(", "));
        if let Ok(g) = GGeometry::new_from_wkt(&wkt) {
            if !g.is_empty().unwrap_or(true) {
                parts.push(g);
            }
        }
    }

    if parts.is_empty() {
        vec![Clone::clone(edge)]
    } else {
        parts
    }
}

/// Consolidate nearby nodes using hierarchical clustering.
///
/// Replaces clusters of nodes within `tolerance` distance with a single
/// weighted centroid node, generating "spider" geometry to maintain
/// connectivity.
///
/// Mirrors Python `consolidate_nodes()`.
pub fn consolidate_nodes(
    geometries: &[GGeometry],
    statuses: &[EdgeStatus],
    tolerance: f64,
    preserve_ends: bool,
) -> (Vec<GGeometry>, Vec<EdgeStatus>) {
    let (node_coords, degrees) = nodes_from_edges(geometries);

    if node_coords.len() < 2 {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // Filter: exclude degree-1 nodes if preserve_ends
    let candidate_indices: Vec<usize> = (0..node_coords.len())
        .filter(|&i| !preserve_ends || degrees[i] > 1)
        .collect();

    if candidate_indices.len() < 2 {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // DBSCAN-like pre-filter: build R-tree of candidate nodes and find
    // connected components of the proximity graph (nodes within tolerance).
    // This avoids O(n²) distance matrix on ALL nodes.
    let points: Vec<GeomWithData<[f64; 2], usize>> = candidate_indices
        .iter()
        .map(|&i| GeomWithData::new(node_coords[i], i))
        .collect();
    let point_tree: RTree<GeomWithData<[f64; 2], usize>> = RTree::bulk_load(points);

    // Build proximity graph
    let n_cand = candidate_indices.len();
    let mut cand_graph = UnGraph::<(), ()>::new_undirected();
    let cand_nodes: Vec<_> = (0..n_cand).map(|_| cand_graph.add_node(())).collect();

    // Map node_coord index → candidate index
    let mut idx_to_cand: HashMap<usize, usize> = HashMap::new();
    for (ci, &ni) in candidate_indices.iter().enumerate() {
        idx_to_cand.insert(ni, ci);
    }

    for (ci, &ni) in candidate_indices.iter().enumerate() {
        let pt = node_coords[ni];
        let env = AABB::from_corners(
            [pt[0] - tolerance, pt[1] - tolerance],
            [pt[0] + tolerance, pt[1] + tolerance],
        );
        for neighbor in point_tree.locate_in_envelope_intersecting(&env) {
            let nj = neighbor.data;
            if nj <= ni {
                continue;
            }
            let dx = node_coords[nj][0] - pt[0];
            let dy = node_coords[nj][1] - pt[1];
            if (dx * dx + dy * dy).sqrt() <= tolerance {
                if let Some(&cj) = idx_to_cand.get(&nj) {
                    cand_graph.add_edge(cand_nodes[ci], cand_nodes[cj], ());
                }
            }
        }
    }

    // Find connected components with >1 node
    let mut visited = vec![false; n_cand];
    let mut proximity_components: Vec<Vec<usize>> = Vec::new();

    for start in 0..n_cand {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut component = Vec::new();
        while let Some(ci) = stack.pop() {
            if visited[ci] {
                continue;
            }
            visited[ci] = true;
            component.push(ci);
            for neighbor in cand_graph.neighbors(cand_nodes[ci]) {
                let nci = neighbor.index();
                if !visited[nci] {
                    stack.push(nci);
                }
            }
        }
        if component.len() > 1 {
            proximity_components.push(component);
        }
    }

    if proximity_components.is_empty() {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // For each proximity component, run hierarchical clustering and collect
    // (centroid, cookie) pairs for clusters with >1 node.
    let mut cluster_info: Vec<([f64; 2], GGeometry)> = Vec::new();

    for component in &proximity_components {
        let comp_node_indices: Vec<usize> =
            component.iter().map(|&ci| candidate_indices[ci]).collect();
        let n = comp_node_indices.len();
        if n < 2 {
            continue;
        }

        // Build condensed distance matrix for this component only
        let mut condensed = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                let ci = comp_node_indices[i];
                let cj = comp_node_indices[j];
                let dx = node_coords[ci][0] - node_coords[cj][0];
                let dy = node_coords[ci][1] - node_coords[cj][1];
                condensed.push((dx * dx + dy * dy).sqrt() as f32);
            }
        }

        let dendrogram = kodama::linkage(&mut condensed, n, kodama::Method::Average);
        let labels = fcluster(&dendrogram, tolerance as f32, n);

        // Group by label
        let mut label_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            label_groups
                .entry(label)
                .or_default()
                .push(comp_node_indices[i]);
        }

        for group in label_groups.values() {
            if group.len() < 2 {
                continue;
            }

            // Compute centroid (mean of coordinates)
            let mut cx = 0.0;
            let mut cy = 0.0;
            for &ni in group {
                cx += node_coords[ni][0];
                cy += node_coords[ni][1];
            }
            cx /= group.len() as f64;
            cy /= group.len() as f64;

            // Build "cookie": convex hull of cluster points, buffered by tolerance/2
            let pts_wkt = format!(
                "MULTIPOINT ({})",
                group
                    .iter()
                    .map(|&ni| format!("{} {}", node_coords[ni][0], node_coords[ni][1]))
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            if let Ok(mp) = GGeometry::new_from_wkt(&pts_wkt) {
                if let Ok(hull) = mp.convex_hull() {
                    if let Ok(cookie) = hull.buffer(tolerance / 2.0, 8) {
                        cluster_info.push(([cx, cy], cookie));
                    }
                }
            }
        }
    }

    if cluster_info.is_empty() {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // Apply spider geometry: for each cluster, cut lines with the cookie
    // and replace with spider lines to the centroid.
    let geom_tree = crate::spatial::build_rtree(geometries);

    let mut result_geoms: Vec<GGeometry> = geometries.to_vec();
    let mut result_statuses: Vec<EdgeStatus> = statuses.to_vec();
    let mut new_spiders: Vec<GGeometry> = Vec::new();

    for (centroid, cookie) in &cluster_info {
        // Query R-tree for candidate edges near this cookie
        let candidates = envelope_query_indices(&geom_tree, cookie);

        let cookie_boundary = match cookie.boundary() {
            Ok(b) => b,
            Err(_) => continue,
        };

        for idx in candidates {
            let geom = &result_geoms[idx];

            if !geom.intersects(cookie).unwrap_or(false) {
                continue;
            }

            // Get intersection with cookie boundary → boundary points
            let pts_geom = match geom.intersection(&cookie_boundary) {
                Ok(p) => p,
                Err(_) => continue,
            };
            if pts_geom.is_empty().unwrap_or(true) {
                continue;
            }

            let coords = extract_point_coords(&pts_geom);
            if coords.is_empty() {
                continue;
            }

            // Cut line with cookie
            if let Ok(remaining) = geom.difference(cookie) {
                result_geoms[idx] = remaining;
                result_statuses[idx] = EdgeStatus::Changed;

                // Create spider lines from boundary points to centroid
                for coord in &coords {
                    let spider_wkt = format!(
                        "LINESTRING ({} {}, {} {})",
                        coord[0], coord[1], centroid[0], centroid[1]
                    );
                    if let Ok(spider) = GGeometry::new_from_wkt(&spider_wkt) {
                        new_spiders.push(spider);
                    }
                }
            }
        }
    }

    // Add spiders as new edges
    for spider in new_spiders {
        result_geoms.push(spider);
        result_statuses.push(EdgeStatus::New);
    }

    // Remove empty geometries and explode MultiLineStrings
    let mut final_geoms = Vec::new();
    let mut final_statuses = Vec::new();

    for (geom, status) in result_geoms.iter().zip(result_statuses.iter()) {
        if geom.is_empty().unwrap_or(true) {
            continue;
        }
        match geom.geometry_type() {
            GeometryTypes::MultiLineString => {
                let n = geom.get_num_geometries().unwrap_or(0);
                for i in 0..n {
                    if let Ok(part) = geom.get_geometry_n(i) {
                        if !part.is_empty().unwrap_or(true) {
                            final_geoms.push(Geom::clone(&part));
                            final_statuses.push(*status);
                        }
                    }
                }
            }
            _ => {
                final_geoms.push(Clone::clone(geom));
                final_statuses.push(*status);
            }
        }
    }

    // Remove interstitial nodes to clean up topology
    remove_interstitial_nodes(&final_geoms, &final_statuses)
}

/// Query an R-tree for geometry indices whose envelopes intersect a given geometry.
/// Public variant for use from other modules.
pub fn envelope_query_indices_pub(
    tree: &rstar::RTree<crate::spatial::IndexedEnvelope>,
    geom: &GGeometry,
) -> Vec<usize> {
    envelope_query_indices(tree, geom)
}

/// Query an R-tree for geometry indices whose envelopes intersect a given geometry.
fn envelope_query_indices(
    tree: &rstar::RTree<crate::spatial::IndexedEnvelope>,
    geom: &GGeometry,
) -> Vec<usize> {
    match geometry_bounds(geom) {
        Some((min, max)) => crate::spatial::query_envelope(tree, min, max),
        None => vec![],
    }
}

/// Extract axis-aligned bounding box from any GEOS geometry.
///
/// Handles Point/LineString envelopes (direct coord_seq) and
/// Polygon envelopes (extract exterior ring first).
fn geometry_bounds(geom: &GGeometry) -> Option<([f64; 2], [f64; 2])> {
    let envelope = geom.envelope().ok()?;

    // For Polygon envelopes, we need to get the exterior ring first
    // since get_coord_seq() only works on Point/LineString/LinearRing.
    let cs = if envelope.geometry_type() == GeometryTypes::Polygon {
        let ring = envelope.get_exterior_ring().ok()?;
        ring.get_coord_seq().ok()?
    } else {
        envelope.get_coord_seq().ok()?
    };

    let n = cs.size().ok()?;
    if n == 0 {
        return None;
    }
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for j in 0..n {
        if let (Ok(x), Ok(y)) = (cs.get_x(j), cs.get_y(j)) {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    Some(([min_x, min_y], [max_x, max_y]))
}

/// Extract point coordinates from a GEOS geometry (Point, MultiPoint, or GeometryCollection).
fn extract_point_coords(geom: &GGeometry) -> Vec<[f64; 2]> {
    let mut coords = Vec::new();
    match geom.geometry_type() {
        GeometryTypes::Point => {
            if let Ok(cs) = geom.get_coord_seq() {
                if let (Ok(x), Ok(y)) = (cs.get_x(0), cs.get_y(0)) {
                    coords.push([x, y]);
                }
            }
        }
        GeometryTypes::MultiPoint | GeometryTypes::GeometryCollection => {
            let n = geom.get_num_geometries().unwrap_or(0);
            for i in 0..n {
                if let Ok(part) = geom.get_geometry_n(i) {
                    coords.extend(extract_point_coords(&Geom::clone(&part)));
                }
            }
        }
        GeometryTypes::LineString => {
            // Intersection with boundary can produce LineStrings (overlapping segments)
            // Extract endpoints
            if let Ok(cs) = geom.get_coord_seq() {
                let n = cs.size().unwrap_or(0);
                if n > 0 {
                    if let (Ok(x), Ok(y)) = (cs.get_x(0), cs.get_y(0)) {
                        coords.push([x, y]);
                    }
                }
                if n > 1 {
                    if let (Ok(x), Ok(y)) = (cs.get_x(n - 1), cs.get_y(n - 1)) {
                        coords.push([x, y]);
                    }
                }
            }
        }
        GeometryTypes::MultiLineString => {
            let n = geom.get_num_geometries().unwrap_or(0);
            for i in 0..n {
                if let Ok(part) = geom.get_geometry_n(i) {
                    coords.extend(extract_point_coords(&Geom::clone(&part)));
                }
            }
        }
        _ => {}
    }
    coords
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

    #[test]
    fn test_consolidate_nodes_close_pair() {
        // Two edges with nearby endpoints at (5,0) and (5.5,0) — within tolerance=2
        // Should consolidate to centroid (5.25, 0) via spider geometry,
        // then remove_interstitial_nodes merges through degree-2 nodes,
        // resulting in a single through-line.
        let g1 = make_line(&[[0.0, 0.0], [5.0, 0.0]]);
        let g2 = make_line(&[[5.5, 0.0], [10.0, 0.0]]);
        let statuses = vec![EdgeStatus::Original, EdgeStatus::Original];

        let (result_geoms, result_statuses) =
            consolidate_nodes(&[g1, g2], &statuses, 2.0, false);

        // The two edges get consolidated through spider geometry and then
        // merged by remove_interstitial_nodes into a single through-line.
        assert!(
            !result_geoms.is_empty(),
            "Expected at least 1 edge after consolidation"
        );
        // Total length should be approximately 10 (original combined length)
        let total_len: f64 = result_geoms.iter().map(|g| g.length().unwrap_or(0.0)).sum();
        assert!(
            total_len > 8.0 && total_len < 12.0,
            "Expected total length ~10, got {}",
            total_len
        );
        // At least one edge should be changed (not all original)
        let has_changed = result_statuses
            .iter()
            .any(|s| *s == EdgeStatus::Changed || *s == EdgeStatus::New);
        assert!(has_changed, "Expected changed/new edges after consolidation");
    }

    #[test]
    fn test_consolidate_nodes_far_apart() {
        // Two edges with nodes far apart — no consolidation should happen
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[100.0, 0.0], [200.0, 0.0]]);
        let statuses = vec![EdgeStatus::Original, EdgeStatus::Original];

        let (result_geoms, result_statuses) =
            consolidate_nodes(&[g1, g2], &statuses, 2.0, false);

        // No consolidation — should return same number of edges
        assert_eq!(result_geoms.len(), 2);
        assert!(result_statuses.iter().all(|s| *s == EdgeStatus::Original));
    }

    #[test]
    fn test_consolidate_nodes_preserve_ends() {
        // T-junction: (0,0)-(5,0)-(10,0) with (5,0)-(5,5)
        // Node (5,0) has degree 3, (0,0), (10,0), (5,5) have degree 1
        // With preserve_ends=true and close end nodes, they shouldn't be consolidated
        let g1 = make_line(&[[0.0, 0.0], [5.0, 0.0]]);
        let g2 = make_line(&[[5.0, 0.0], [10.0, 0.0]]);
        let g3 = make_line(&[[5.0, 0.0], [5.0, 5.0]]);
        let statuses = vec![EdgeStatus::Original; 3];

        let (result_geoms, _) =
            consolidate_nodes(&[g1, g2, g3], &statuses, 2.0, true);

        // With preserve_ends=true, degree-1 nodes excluded from clustering
        // Only the degree-3 node remains, but a single node can't form a cluster
        // So nothing should change
        assert!(result_geoms.len() >= 3);
    }

    #[test]
    fn test_snap_n_split_basic() {
        // Line from (0,0) to (10,0), split at (5,0)
        let line = make_line(&[[0.0, 0.0], [10.0, 0.0]]);
        let pt = GGeometry::new_from_wkt("POINT (5 0)").unwrap();
        let parts = snap_n_split(&line, &pt, 1e-4);

        assert_eq!(parts.len(), 2, "Expected 2 parts, got {}", parts.len());
        // First part should be (0,0)→(5,0)
        // Second part should be (5,0)→(10,0)
        let len0 = parts[0].length().unwrap();
        let len1 = parts[1].length().unwrap();
        assert!((len0 - 5.0).abs() < 0.01, "Part 0 length: {}", len0);
        assert!((len1 - 5.0).abs() < 0.01, "Part 1 length: {}", len1);
    }

    #[test]
    fn test_snap_n_split_at_endpoint() {
        // Split at an endpoint should not produce a split
        let line = make_line(&[[0.0, 0.0], [10.0, 0.0]]);
        let pt = GGeometry::new_from_wkt("POINT (0 0)").unwrap();
        let parts = snap_n_split(&line, &pt, 1e-4);
        // Should return original (no split at endpoint)
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn test_induce_nodes_crossing() {
        // Two lines crossing at (5,0): one from (0,0)→(10,0),
        // other from (5,-5)→(5,5) — but the first line doesn't have a
        // vertex at (5,0), so induce_nodes should split it.
        let g1 = make_line(&[[0.0, 0.0], [10.0, 0.0]]);
        let g2 = make_line(&[[5.0, -5.0], [5.0, 0.0]]);
        let statuses = vec![EdgeStatus::Original; 2];

        let (result_geoms, result_statuses) = induce_nodes(&[g1, g2], &statuses, 1e-4);

        // g1 should be split at (5,0), giving 3 total edges
        assert!(
            result_geoms.len() >= 3,
            "Expected at least 3 edges after inducing node, got {}",
            result_geoms.len()
        );
    }

    #[test]
    fn test_extract_point_coords_point() {
        let pt = GGeometry::new_from_wkt("POINT (3.5 7.2)").unwrap();
        let coords = extract_point_coords(&pt);
        assert_eq!(coords.len(), 1);
        assert!((coords[0][0] - 3.5).abs() < 1e-10);
        assert!((coords[0][1] - 7.2).abs() < 1e-10);
    }

    #[test]
    fn test_extract_point_coords_multipoint() {
        let mp = GGeometry::new_from_wkt("MULTIPOINT (1 2, 3 4, 5 6)").unwrap();
        let coords = extract_point_coords(&mp);
        assert_eq!(coords.len(), 3);
    }
}
