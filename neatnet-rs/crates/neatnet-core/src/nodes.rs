//! Node operations: topology fixing, node consolidation, component labeling.
//!
//! Ports Python `neatnet.nodes`: `fix_topology`, `consolidate_nodes`,
//! `get_components`, `remove_interstitial_nodes`, `induce_nodes`.

use std::collections::{BTreeMap, HashMap};

use geo::{BooleanOps, BoundingRect, Buffer, ConvexHull, Distance, Euclidean, Intersects};
use geo_types::{Coord, LineString, MultiLineString, Point, Polygon};
use petgraph::graph::UnGraph;
use rstar::primitives::GeomWithData;
use rstar::{RTree, AABB};

use crate::ops;
use crate::types::EdgeStatus;

/// Rounded coordinate key for endpoint hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
pub fn nodes_from_edges(
    geometries: &[LineString<f64>],
) -> (Vec<[f64; 2]>, Vec<usize>) {
    let mut node_counts: HashMap<NodeKey, (f64, f64, usize)> = HashMap::new();

    for geom in geometries {
        let coords = &geom.0;
        if coords.len() < 2 { continue; }
        // Start point
        let c0 = coords[0];
        let key = node_key(c0.x, c0.y);
        let entry = node_counts.entry(key).or_insert((c0.x, c0.y, 0));
        entry.2 += 1;
        // End point
        let cn = coords[coords.len() - 1];
        let key = node_key(cn.x, cn.y);
        let entry = node_counts.entry(key).or_insert((cn.x, cn.y, 0));
        entry.2 += 1;
    }

    // Sort by key for deterministic ordering (HashMap iteration is non-deterministic)
    let mut sorted_nodes: Vec<_> = node_counts.into_iter().collect();
    sorted_nodes.sort_by_key(|(k, _)| *k);

    let mut coords_out = Vec::with_capacity(sorted_nodes.len());
    let mut degrees = Vec::with_capacity(sorted_nodes.len());
    for (_, (x, y, deg)) in &sorted_nodes {
        coords_out.push([*x, *y]);
        degrees.push(*deg);
    }
    (coords_out, degrees)
}

/// Determine component labels for chains of edges linked by degree-2 nodes.
pub fn get_components(geometries: &[LineString<f64>]) -> Vec<usize> {
    let n = geometries.len();
    if n == 0 { return vec![]; }

    let mut node_to_edges: HashMap<NodeKey, Vec<usize>> = HashMap::new();
    for (edge_idx, geom) in geometries.iter().enumerate() {
        let coords = &geom.0;
        if coords.len() < 2 { continue; }
        let c0 = coords[0];
        node_to_edges.entry(node_key(c0.x, c0.y)).or_default().push(edge_idx);
        let cn = coords[coords.len() - 1];
        node_to_edges.entry(node_key(cn.x, cn.y)).or_default().push(edge_idx);
    }

    let is_closed: Vec<bool> = geometries.iter().map(|g| {
        g.0.len() >= 4 && g.0[0] == g.0[g.0.len() - 1]
    }).collect();

    let mut graph = UnGraph::<(), ()>::new_undirected();
    let graph_nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    for (_key, edge_indices) in &node_to_edges {
        if edge_indices.len() != 2 { continue; }
        let e0 = edge_indices[0];
        let e1 = edge_indices[1];
        if is_closed[e0] || is_closed[e1] { continue; }
        graph.add_edge(graph_nodes[e0], graph_nodes[e1], ());
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
            for neighbor in graph.neighbors(graph_nodes[node]) {
                let nidx = neighbor.index();
                if !visited[nidx] { stack.push(nidx); }
            }
        }
        label += 1;
    }

    labels
}

/// Remove interstitial (degree-2) nodes by merging edge chains.
pub fn remove_interstitial_nodes(
    geometries: &[LineString<f64>],
    statuses: &[EdgeStatus],
) -> (Vec<LineString<f64>>, Vec<EdgeStatus>) {
    if geometries.len() < 2 {
        return (geometries.to_vec(), statuses.to_vec());
    }

    let labels = get_components(geometries);

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        groups.entry(label).or_default().push(idx);
    }

    let mut result_geoms = Vec::new();
    let mut result_statuses = Vec::new();

    let mut sorted_labels: Vec<_> = groups.keys().copied().collect();
    sorted_labels.sort();
    for label in sorted_labels {
        let indices = &groups[&label];
        if indices.len() == 1 {
            result_geoms.push(geometries[indices[0]].clone());
            result_statuses.push(statuses[indices[0]]);
        } else {
            let group_statuses: Vec<EdgeStatus> =
                indices.iter().map(|&i| statuses[i]).collect();
            let merged_status = EdgeStatus::aggregate(&group_statuses);

            let group_geoms: Vec<LineString<f64>> =
                indices.iter().map(|&i| geometries[i].clone()).collect();
            let merged = ops::line_merge(&group_geoms);

            if merged.len() == 1 {
                result_geoms.push(merged.into_iter().next().unwrap());
                result_statuses.push(merged_status);
            } else {
                // Fallback: keep individual geometries
                for &idx in indices {
                    result_geoms.push(geometries[idx].clone());
                    result_statuses.push(statuses[idx]);
                }
            }
        }
    }

    (result_geoms, result_statuses)
}

/// Fix street network topology.
pub fn fix_topology(
    geometries: &[LineString<f64>],
    statuses: &[EdgeStatus],
    eps: f64,
) -> (Vec<LineString<f64>>, Vec<EdgeStatus>) {
    // Step 1: Remove duplicates (by normalized WKT)
    let mut seen = std::collections::HashSet::new();
    let mut deduped_geoms = Vec::new();
    let mut deduped_statuses = Vec::new();

    for (geom, &status) in geometries.iter().zip(statuses.iter()) {
        let normalized = ops::normalize_linestring(geom);
        let wkt = ops::linestring_to_wkt(&normalized);
        if seen.insert(wkt) {
            deduped_geoms.push(geom.clone());
            deduped_statuses.push(status);
        }
    }

    // Step 2: Induce nodes at intersections
    let (induced_geoms, induced_statuses) = induce_nodes(&deduped_geoms, &deduped_statuses, eps);

    // Step 3: Remove interstitial nodes
    remove_interstitial_nodes(&induced_geoms, &induced_statuses)
}

/// Add missing nodes where line endpoints intersect other edges.
pub fn induce_nodes(
    geometries: &[LineString<f64>],
    statuses: &[EdgeStatus],
    eps: f64,
) -> (Vec<LineString<f64>>, Vec<EdgeStatus>) {
    if geometries.is_empty() {
        return (vec![], vec![]);
    }

    let mismatch_points = identify_degree_mismatch(geometries, eps);
    let (loop_non_loop_pts, loop_loop_pts) = makes_loop_contact(geometries, eps);

    let mut all_split_points: Vec<Coord<f64>> = Vec::new();
    let mut seen_keys = std::collections::HashSet::new();

    for pt in mismatch_points.iter()
        .chain(loop_non_loop_pts.iter())
        .chain(loop_loop_pts.iter())
    {
        let key = node_key(pt.x, pt.y);
        if seen_keys.insert(key) {
            all_split_points.push(*pt);
        }
    }

    if all_split_points.is_empty() {
        return (geometries.to_vec(), statuses.to_vec());
    }

    split_edges_at_points(geometries, statuses, &all_split_points, eps)
}

fn identify_degree_mismatch(geometries: &[LineString<f64>], eps: f64) -> Vec<Coord<f64>> {
    let (node_coords, degrees) = nodes_from_edges(geometries);
    let geom_tree = crate::spatial::build_rtree(geometries);

    let mut result = Vec::new();
    for (i, coord) in node_coords.iter().enumerate() {
        let candidates = crate::spatial::query_envelope(
            &geom_tree,
            [coord[0] - eps, coord[1] - eps],
            [coord[0] + eps, coord[1] + eps],
        );

        let pt = Point::new(coord[0], coord[1]);
        let mut expected = 0usize;
        for &idx in &candidates {
            let dist = Euclidean.distance(&pt, &geometries[idx]);
            if dist <= eps {
                expected += 1;
            }
        }

        if expected != degrees[i] {
            result.push(Coord { x: coord[0], y: coord[1] });
        }
    }
    result
}

fn makes_loop_contact(
    geometries: &[LineString<f64>],
    eps: f64,
) -> (Vec<Coord<f64>>, Vec<Coord<f64>>) {
    let mut loop_indices = Vec::new();
    let mut non_loop_indices = Vec::new();

    for (i, geom) in geometries.iter().enumerate() {
        if geom.0.len() >= 4 && geom.0[0] == geom.0[geom.0.len() - 1] {
            loop_indices.push(i);
        } else {
            non_loop_indices.push(i);
        }
    }

    if loop_indices.is_empty() {
        return (vec![], vec![]);
    }

    let non_loop_geoms: Vec<LineString<f64>> = non_loop_indices.iter().map(|&i| geometries[i].clone()).collect();
    let non_loop_tree = crate::spatial::build_rtree(&non_loop_geoms);

    let loop_geoms: Vec<LineString<f64>> = loop_indices.iter().map(|&i| geometries[i].clone()).collect();
    let loop_tree = crate::spatial::build_rtree(&loop_geoms);

    let mut non_loop_contact = Vec::new();
    let mut loop_contact = Vec::new();

    for &li in &loop_indices {
        let coords = &geometries[li].0;
        for vi in 0..coords.len() {
            let c = coords[vi];
            let pt = Point::new(c.x, c.y);

            // Check non-loop contact
            let nl_candidates = crate::spatial::query_envelope(
                &non_loop_tree,
                [c.x - eps, c.y - eps],
                [c.x + eps, c.y + eps],
            );
            for &idx in &nl_candidates {
                let dist = Euclidean.distance(&pt, &non_loop_geoms[idx]);
                if dist <= eps {
                    non_loop_contact.push(c);
                    break;
                }
            }

            // Check loop-loop contact
            let l_candidates = crate::spatial::query_envelope(
                &loop_tree,
                [c.x - eps, c.y - eps],
                [c.x + eps, c.y + eps],
            );
            let mut touch_count = 0;
            for &idx in &l_candidates {
                let dist = Euclidean.distance(&pt, &loop_geoms[idx]);
                if dist <= eps {
                    touch_count += 1;
                }
            }
            if touch_count > 1 {
                loop_contact.push(c);
            }
        }
    }

    (non_loop_contact, loop_contact)
}

fn split_edges_at_points(
    geometries: &[LineString<f64>],
    statuses: &[EdgeStatus],
    split_points: &[Coord<f64>],
    eps: f64,
) -> (Vec<LineString<f64>>, Vec<EdgeStatus>) {
    let mut result_geoms: Vec<LineString<f64>> = geometries.to_vec();
    let mut result_statuses: Vec<EdgeStatus> = statuses.to_vec();

    for split_pt in split_points {
        let tree = crate::spatial::build_rtree(&result_geoms);
        let candidates = crate::spatial::query_envelope(
            &tree,
            [split_pt.x - eps * 10.0, split_pt.y - eps * 10.0],
            [split_pt.x + eps * 10.0, split_pt.y + eps * 10.0],
        );

        let mut to_remove = Vec::new();
        let mut to_add_geoms = Vec::new();
        let mut to_add_statuses = Vec::new();

        let pt = Point::new(split_pt.x, split_pt.y);
        for &idx in &candidates {
            let dist = Euclidean.distance(&pt, &result_geoms[idx]);
            if dist > eps { continue; }

            let parts = snap_n_split(&result_geoms[idx], *split_pt, eps);
            if parts.len() > 1 {
                to_remove.push(idx);
                for part in parts {
                    to_add_geoms.push(part);
                    to_add_statuses.push(EdgeStatus::Changed);
                }
            }
        }

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
fn snap_n_split(edge: &LineString<f64>, point: Coord<f64>, eps: f64) -> Vec<LineString<f64>> {
    let coords = &edge.0;
    if coords.len() < 2 {
        return vec![edge.clone()];
    }

    // Find the closest vertex to the split point
    let mut split_idx = None;
    let mut best_dist = eps * 2.0;

    // First, snap: insert the point as a vertex if it's close to a segment
    let mut snapped_coords = coords.to_vec();

    for i in 0..snapped_coords.len() {
        let dx = snapped_coords[i].x - point.x;
        let dy = snapped_coords[i].y - point.y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < best_dist {
            best_dist = dist;
            if i > 0 && i < snapped_coords.len() - 1 {
                split_idx = Some(i);
            }
        }
    }

    // If no vertex is close enough, project point onto segments
    if split_idx.is_none() {
        for i in 0..snapped_coords.len() - 1 {
            let a = snapped_coords[i];
            let b = snapped_coords[i + 1];
            let dx = b.x - a.x;
            let dy = b.y - a.y;
            let len_sq = dx * dx + dy * dy;
            if len_sq < 1e-20 { continue; }

            let t = ((point.x - a.x) * dx + (point.y - a.y) * dy) / len_sq;
            if t <= 0.0 || t >= 1.0 { continue; }

            let proj = Coord { x: a.x + t * dx, y: a.y + t * dy };
            let dist = ((proj.x - point.x).powi(2) + (proj.y - point.y).powi(2)).sqrt();
            if dist < eps * 2.0 {
                // Insert the point
                snapped_coords.insert(i + 1, point);
                split_idx = Some(i + 1);
                break;
            }
        }
    }

    let Some(idx) = split_idx else {
        return vec![edge.clone()];
    };

    // Build two sub-lines
    let mut parts = Vec::new();
    if idx >= 1 {
        let part1: Vec<Coord<f64>> = snapped_coords[..=idx].to_vec();
        if part1.len() >= 2 {
            parts.push(LineString::new(part1));
        }
    }
    if idx < snapped_coords.len() - 1 {
        let part2: Vec<Coord<f64>> = snapped_coords[idx..].to_vec();
        if part2.len() >= 2 {
            parts.push(LineString::new(part2));
        }
    }

    if parts.is_empty() {
        vec![edge.clone()]
    } else {
        parts
    }
}

/// Consolidate nearby nodes using hierarchical clustering.
pub fn consolidate_nodes(
    geometries: &[LineString<f64>],
    statuses: &[EdgeStatus],
    tolerance: f64,
    preserve_ends: bool,
) -> (Vec<LineString<f64>>, Vec<EdgeStatus>) {
    let (node_coords, degrees) = nodes_from_edges(geometries);

    if node_coords.len() < 2 {
        return (geometries.to_vec(), statuses.to_vec());
    }

    let candidate_indices: Vec<usize> = (0..node_coords.len())
        .filter(|&i| !preserve_ends || degrees[i] > 1)
        .collect();

    if candidate_indices.len() < 2 {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // Build R-tree and proximity graph
    let points: Vec<GeomWithData<[f64; 2], usize>> = candidate_indices
        .iter()
        .map(|&i| GeomWithData::new(node_coords[i], i))
        .collect();
    let point_tree: RTree<GeomWithData<[f64; 2], usize>> = RTree::bulk_load(points);

    let n_cand = candidate_indices.len();
    let mut cand_graph = UnGraph::<(), ()>::new_undirected();
    let cand_nodes: Vec<_> = (0..n_cand).map(|_| cand_graph.add_node(())).collect();

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
            if nj <= ni { continue; }
            let dx = node_coords[nj][0] - pt[0];
            let dy = node_coords[nj][1] - pt[1];
            if (dx * dx + dy * dy).sqrt() <= tolerance {
                if let Some(&cj) = idx_to_cand.get(&nj) {
                    cand_graph.add_edge(cand_nodes[ci], cand_nodes[cj], ());
                }
            }
        }
    }

    // Find proximity components with >1 node
    let mut visited = vec![false; n_cand];
    let mut proximity_components: Vec<Vec<usize>> = Vec::new();

    for start in 0..n_cand {
        if visited[start] { continue; }
        let mut stack = vec![start];
        let mut component = Vec::new();
        while let Some(ci) = stack.pop() {
            if visited[ci] { continue; }
            visited[ci] = true;
            component.push(ci);
            for neighbor in cand_graph.neighbors(cand_nodes[ci]) {
                let nci = neighbor.index();
                if !visited[nci] { stack.push(nci); }
            }
        }
        if component.len() > 1 {
            proximity_components.push(component);
        }
    }

    if proximity_components.is_empty() {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // Hierarchical clustering per proximity component
    let mut cluster_info: Vec<([f64; 2], Polygon<f64>)> = Vec::new();

    for component in &proximity_components {
        let comp_node_indices: Vec<usize> =
            component.iter().map(|&ci| candidate_indices[ci]).collect();
        let n = comp_node_indices.len();
        if n < 2 { continue; }

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

        let mut label_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, &label) in labels.iter().enumerate() {
            label_groups.entry(label).or_default().push(comp_node_indices[i]);
        }

        for group in label_groups.values() {
            if group.len() < 2 { continue; }

            let mut cx = 0.0;
            let mut cy = 0.0;
            for &ni in group {
                cx += node_coords[ni][0];
                cy += node_coords[ni][1];
            }
            cx /= group.len() as f64;
            cy /= group.len() as f64;

            // Build cookie: convex hull of points, buffered by tolerance/2
            let mp_coords: Vec<Coord<f64>> = group.iter()
                .map(|&ni| Coord { x: node_coords[ni][0], y: node_coords[ni][1] })
                .collect();

            let hull_line = LineString::new(mp_coords.clone());
            let hull_poly = hull_line.convex_hull();
            // geo::Buffer can't handle degenerate (zero-area) polygons,
            // so buffer the LineString directly in that case
            let cookie_mp = if geo::Area::unsigned_area(&hull_poly) < 1e-10 {
                hull_line.buffer(tolerance / 2.0)
            } else {
                hull_poly.buffer(tolerance / 2.0)
            };
            // Buffer returns MultiPolygon; take the first polygon
            if let Some(cookie) = cookie_mp.0.into_iter().next() {
                cluster_info.push(([cx, cy], cookie));
            }
        }
    }

    if cluster_info.is_empty() {
        return (geometries.to_vec(), statuses.to_vec());
    }

    // Apply spider geometry
    let geom_tree = crate::spatial::build_rtree(geometries);

    let mut result_geoms: Vec<LineString<f64>> = geometries.to_vec();
    let mut result_statuses: Vec<EdgeStatus> = statuses.to_vec();
    let mut new_spiders: Vec<LineString<f64>> = Vec::new();

    for (centroid, cookie) in &cluster_info {
        let candidates = envelope_query_indices(&geom_tree, cookie);

        let cookie_boundary = cookie.exterior().clone();

        for idx in candidates {
            let geom = &result_geoms[idx];

            if !geom.intersects(cookie) {
                continue;
            }

            // Get intersection with cookie boundary → boundary points
            let coords = extract_boundary_intersection_points(geom, &cookie_boundary);
            if coords.is_empty() { continue; }

            // Cut line with cookie: use BooleanOps clip
            let mls = MultiLineString::new(vec![geom.clone()]);
            let diff_mls = cookie.clip(&mls, true);
            // diff_mls is the part outside the cookie

            let diff_lines: Vec<LineString<f64>> = diff_mls.0;
            if diff_lines.is_empty() {
                // Entire line was inside cookie
                result_geoms[idx] = LineString::new(vec![]);
                result_statuses[idx] = EdgeStatus::Changed;
            } else if diff_lines.len() == 1 {
                result_geoms[idx] = diff_lines.into_iter().next().unwrap();
                result_statuses[idx] = EdgeStatus::Changed;
            } else {
                // Multiple parts: keep as separate geometries later
                result_geoms[idx] = diff_lines[0].clone();
                result_statuses[idx] = EdgeStatus::Changed;
                for part in &diff_lines[1..] {
                    new_spiders.push(part.clone());
                }
            }

            // Create spider lines from boundary points to centroid
            for coord in &coords {
                let spider = LineString::new(vec![
                    Coord { x: coord[0], y: coord[1] },
                    Coord { x: centroid[0], y: centroid[1] },
                ]);
                new_spiders.push(spider);
            }
        }
    }

    // Add spiders and remove empty geometries
    for spider in new_spiders {
        result_geoms.push(spider);
        result_statuses.push(EdgeStatus::New);
    }

    let mut final_geoms = Vec::new();
    let mut final_statuses = Vec::new();

    for (geom, status) in result_geoms.iter().zip(result_statuses.iter()) {
        if geom.0.len() >= 2 {
            final_geoms.push(geom.clone());
            final_statuses.push(*status);
        }
    }

    remove_interstitial_nodes(&final_geoms, &final_statuses)
}

/// Query R-tree for line indices near a polygon's bounding box.
pub fn envelope_query_indices_pub(
    tree: &rstar::RTree<crate::spatial::IndexedEnvelope>,
    poly: &Polygon<f64>,
) -> Vec<usize> {
    envelope_query_indices(tree, poly)
}

fn envelope_query_indices(
    tree: &rstar::RTree<crate::spatial::IndexedEnvelope>,
    poly: &Polygon<f64>,
) -> Vec<usize> {
    match poly.bounding_rect() {
        Some(rect) => crate::spatial::query_envelope(
            tree,
            [rect.min().x, rect.min().y],
            [rect.max().x, rect.max().y],
        ),
        None => vec![],
    }
}

/// Extract points where a line intersects a ring (polygon exterior).
fn extract_boundary_intersection_points(
    line: &LineString<f64>,
    ring: &LineString<f64>,
) -> Vec<[f64; 2]> {
    use geo::line_intersection::{line_intersection, LineIntersection};
    use geo_types::Line;

    let mut coords = Vec::new();
    for seg_a in line.0.windows(2) {
        let la = Line::new(seg_a[0], seg_a[1]);
        for seg_b in ring.0.windows(2) {
            let lb = Line::new(seg_b[0], seg_b[1]);
            if let Some(inter) = line_intersection(la, lb) {
                match inter {
                    LineIntersection::SinglePoint { intersection, .. } => {
                        coords.push([intersection.x, intersection.y]);
                    }
                    LineIntersection::Collinear { intersection } => {
                        coords.push([intersection.start.x, intersection.start.y]);
                        coords.push([intersection.end.x, intersection.end.y]);
                    }
                }
            }
        }
    }
    // Deduplicate
    let mut seen = std::collections::HashSet::new();
    coords.retain(|c| {
        let key = ((c[0] * 1e6).round() as i64, (c[1] * 1e6).round() as i64);
        seen.insert(key)
    });
    coords
}

fn fcluster(dendrogram: &kodama::Dendrogram<f32>, threshold: f32, n: usize) -> Vec<usize> {
    let mut labels = vec![0usize; n];
    let mut next_label = 0;
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
            let a = find(&mut parent, step.cluster1);
            let b = find(&mut parent, step.cluster2);
            parent[a] = new_cluster;
            parent[b] = new_cluster;
            parent[new_cluster] = new_cluster;
        } else {
            parent[new_cluster] = new_cluster;
        }
    }

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
    use geo::Length;
    use geo_types::Coord;

    fn make_line(coords: &[[f64; 2]]) -> LineString<f64> {
        LineString::new(coords.iter().map(|c| Coord { x: c[0], y: c[1] }).collect())
    }

    #[test]
    fn test_nodes_from_edges() {
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let (coords, degrees) = nodes_from_edges(&[g1, g2]);
        assert_eq!(coords.len(), 3);
        assert_eq!(degrees.len(), 3);
    }

    #[test]
    fn test_get_components_chain() {
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let g3 = make_line(&[[2.0, 0.0], [3.0, 0.0]]);
        let labels = get_components(&[g1, g2, g3]);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
    }

    #[test]
    fn test_get_components_branch() {
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[1.0, 0.0], [2.0, 0.0]]);
        let g3 = make_line(&[[1.0, 0.0], [1.0, 1.0]]);
        let labels = get_components(&[g1, g2, g3]);
        assert!(labels[0] != labels[2] || labels[1] != labels[2]);
    }

    #[test]
    fn test_fcluster_basic() {
        let mut condensed = vec![1.0f32, 10.0, 11.0, 9.0, 10.0, 1.0];
        let dendro = kodama::linkage(&mut condensed, 4, kodama::Method::Average);
        let labels = fcluster(&dendro, 2.0, 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn test_consolidate_nodes_close_pair() {
        let g1 = make_line(&[[0.0, 0.0], [5.0, 0.0]]);
        let g2 = make_line(&[[5.5, 0.0], [10.0, 0.0]]);
        let statuses = vec![EdgeStatus::Original, EdgeStatus::Original];
        let (result_geoms, result_statuses) =
            consolidate_nodes(&[g1, g2], &statuses, 2.0, false);
        assert!(!result_geoms.is_empty());
        let total_len: f64 = result_geoms.iter().map(|g| Euclidean.length(g)).sum();
        assert!(total_len > 8.0 && total_len < 12.0, "Expected total length ~10, got {}", total_len);
        let has_changed = result_statuses.iter().any(|s| *s == EdgeStatus::Changed || *s == EdgeStatus::New);
        assert!(has_changed);
    }

    #[test]
    fn test_consolidate_nodes_far_apart() {
        let g1 = make_line(&[[0.0, 0.0], [1.0, 0.0]]);
        let g2 = make_line(&[[100.0, 0.0], [200.0, 0.0]]);
        let statuses = vec![EdgeStatus::Original, EdgeStatus::Original];
        let (result_geoms, result_statuses) =
            consolidate_nodes(&[g1, g2], &statuses, 2.0, false);
        assert_eq!(result_geoms.len(), 2);
        assert!(result_statuses.iter().all(|s| *s == EdgeStatus::Original));
    }

    #[test]
    fn test_consolidate_nodes_preserve_ends() {
        let g1 = make_line(&[[0.0, 0.0], [5.0, 0.0]]);
        let g2 = make_line(&[[5.0, 0.0], [10.0, 0.0]]);
        let g3 = make_line(&[[5.0, 0.0], [5.0, 5.0]]);
        let statuses = vec![EdgeStatus::Original; 3];
        let (result_geoms, _) = consolidate_nodes(&[g1, g2, g3], &statuses, 2.0, true);
        assert!(result_geoms.len() >= 3);
    }

    #[test]
    fn test_buffer_degenerate_hull_fallback() {
        // Two collinear points produce a degenerate convex hull (zero area).
        // geo::Buffer can't buffer zero-area polygons, so we fall back to
        // buffering the LineString directly.
        let mp_coords = vec![
            Coord { x: 5.0, y: 0.0 },
            Coord { x: 5.5, y: 0.0 },
        ];
        let hull_line = LineString::new(mp_coords);
        let hull_poly = hull_line.convex_hull();
        // Degenerate hull has zero area
        assert!(geo::Area::unsigned_area(&hull_poly) < 1e-10);
        // Buffering degenerate polygon returns empty
        let poly_buf = hull_poly.buffer(1.0);
        assert!(poly_buf.0.is_empty(), "geo::Buffer of degenerate polygon returns empty");
        // But buffering the LineString works
        let line_buf = hull_line.buffer(1.0);
        assert!(!line_buf.0.is_empty(), "LineString buffer should work");
        assert!(geo::Area::unsigned_area(&line_buf.0[0]) > 0.0);
    }

    #[test]
    fn test_snap_n_split_basic() {
        let line = make_line(&[[0.0, 0.0], [10.0, 0.0]]);
        let pt = Coord { x: 5.0, y: 0.0 };
        let parts = snap_n_split(&line, pt, 1e-4);
        assert_eq!(parts.len(), 2, "Expected 2 parts, got {}", parts.len());
        let len0 = Euclidean.length(&parts[0]);
        let len1 = Euclidean.length(&parts[1]);
        assert!((len0 - 5.0).abs() < 0.01, "Part 0 length: {}", len0);
        assert!((len1 - 5.0).abs() < 0.01, "Part 1 length: {}", len1);
    }

    #[test]
    fn test_snap_n_split_at_endpoint() {
        let line = make_line(&[[0.0, 0.0], [10.0, 0.0]]);
        let pt = Coord { x: 0.0, y: 0.0 };
        let parts = snap_n_split(&line, pt, 1e-4);
        assert_eq!(parts.len(), 1);
    }

    #[test]
    fn test_induce_nodes_crossing() {
        let g1 = make_line(&[[0.0, 0.0], [10.0, 0.0]]);
        let g2 = make_line(&[[5.0, -5.0], [5.0, 0.0]]);
        let statuses = vec![EdgeStatus::Original; 2];
        let (result_geoms, _) = induce_nodes(&[g1, g2], &statuses, 1e-4);
        assert!(result_geoms.len() >= 3, "Expected at least 3 edges, got {}", result_geoms.len());
    }
}
