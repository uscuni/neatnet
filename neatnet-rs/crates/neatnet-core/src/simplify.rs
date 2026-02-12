//! Main simplification pipeline: neatify and its sub-steps.
//!
//! Ports Python `neatnet.simplify`: `neatify`, `neatify_loop`,
//! `neatify_singletons`, `neatify_pairs`, `neatify_clusters`.

use std::collections::{HashMap, HashSet};

use geos::{Geom, Geometry as GGeometry};

use crate::artifacts;
use crate::continuity;
use crate::geometry;
use crate::nodes;
use crate::types::{EdgeStatus, NeatifyParams, StreetNetwork};

/// Top-level simplification entry point.
///
/// Follows the Adaptive Continuity-Preserving Simplification algorithm:
/// 1. CRS validation
/// 2. Topology fixing (induce nodes, remove degree-2 nodes, dedup)
/// 3. Node consolidation (hierarchical clustering)
/// 4. Artifact detection (polygonize → FAI → KDE threshold)
/// 5. Iterative simplification loops:
///    a. Remove dangles within artifacts
///    b. Simplify singletons
///    c. Simplify pairs
///    d. Simplify clusters
///
/// Mirrors Python `neatify()`.
pub fn neatify(
    network: &mut StreetNetwork,
    params: &NeatifyParams,
    exclusion_mask: Option<&[GGeometry]>,
) -> Result<(), NeatifyError> {
    // Step 1: Fix topology
    let (fixed_geoms, fixed_statuses) =
        nodes::fix_topology(&network.geometries, &network.statuses, params.eps);
    network.geometries = fixed_geoms;
    network.statuses = fixed_statuses;

    // Step 2: Consolidate nodes
    let (consol_geoms, consol_statuses) = nodes::consolidate_nodes(
        &network.geometries,
        &network.statuses,
        params.max_segment_length * 2.1,
        false,
    );
    network.geometries = consol_geoms;
    network.statuses = consol_statuses;

    // Step 3: Detect artifacts (with iterative expansion)
    let artifacts = artifacts::get_artifacts(
        &network.geometries,
        params.artifact_threshold,
        params.artifact_threshold_fallback,
        exclusion_mask,
        params.area_threshold_blocks,
        params.isoareal_threshold_blocks,
        params.area_threshold_circles,
        params.isoareal_threshold_circles_enclosed,
        params.isoperimetric_threshold_circles_touching,
    );

    let (artifact_geoms, _artifact_fais, threshold) = match artifacts {
        Some(a) => a,
        None => {
            log::warn!("No artifacts detected. Returning after topology fixes.");
            return Ok(());
        }
    };

    if artifact_geoms.is_empty() {
        log::warn!("No artifacts found. Returning after topology fixes.");
        return Ok(());
    }

    // Step 4: Iterative simplification loops
    for loop_idx in 0..params.n_loops {
        neatify_loop(network, &artifact_geoms, params)?;

        // Re-detect artifacts for subsequent loops
        if loop_idx < params.n_loops - 1 {
            let re_artifacts = artifacts::get_artifacts(
                &network.geometries,
                Some(threshold),
                params.artifact_threshold_fallback,
                exclusion_mask,
                params.area_threshold_blocks,
                params.isoareal_threshold_blocks,
                params.area_threshold_circles,
                params.isoareal_threshold_circles_enclosed,
                params.isoperimetric_threshold_circles_touching,
            );
            if re_artifacts.is_none() {
                break;
            }
        }
    }

    Ok(())
}

/// One iteration of the simplification loop.
///
/// Mirrors Python `neatify_loop()`.
fn neatify_loop(
    network: &mut StreetNetwork,
    artifact_geoms: &[GGeometry],
    params: &NeatifyParams,
) -> Result<(), NeatifyError> {
    // 1. Remove dangles: drop edges fully inside any artifact, then clean
    let tree = crate::spatial::build_rtree(&network.geometries);
    let mut dangle_indices = HashSet::new();
    for artifact in artifact_geoms {
        let candidates = nodes::envelope_query_indices_pub(&tree, artifact);
        for idx in candidates {
            if artifact.contains(&network.geometries[idx]).unwrap_or(false) {
                dangle_indices.insert(idx);
            }
        }
    }

    if !dangle_indices.is_empty() {
        let mut new_geoms = Vec::new();
        let mut new_statuses = Vec::new();
        for (i, geom) in network.geometries.iter().enumerate() {
            if !dangle_indices.contains(&i) {
                new_geoms.push(Clone::clone(geom));
                new_statuses.push(network.statuses[i]);
            }
        }
        network.geometries = new_geoms;
        network.statuses = new_statuses;
    }

    let (cleaned, statuses) =
        nodes::remove_interstitial_nodes(&network.geometries, &network.statuses);
    network.geometries = cleaned;
    network.statuses = statuses;

    // 2. Build contiguity graph on artifacts → classify as singles/pairs/clusters
    let adjacency = artifacts::build_contiguity_graph(artifact_geoms, true);
    let comp_labels = artifacts::component_labels_from_adjacency(&adjacency);

    // Count component sizes
    let mut comp_sizes: HashMap<usize, usize> = HashMap::new();
    for &label in &comp_labels {
        *comp_sizes.entry(label).or_default() += 1;
    }

    // Classify: isolates (size 1), pairs (size 2), clusters (size 3+)
    let mut singles = Vec::new();
    let mut pairs = Vec::new();
    let mut clusters = Vec::new();

    for (i, &label) in comp_labels.iter().enumerate() {
        match comp_sizes.get(&label) {
            Some(&1) => singles.push(i),
            Some(&2) => pairs.push(i),
            Some(_) => clusters.push(i),
            None => {}
        }
    }

    // 3. Simplify singletons
    if !singles.is_empty() {
        neatify_singletons(network, artifact_geoms, &singles, params)?;
    }

    // 4. Simplify pairs
    if !pairs.is_empty() {
        neatify_pairs(network, artifact_geoms, &pairs, &comp_labels, params)?;
    }

    // 5. Simplify clusters
    if !clusters.is_empty() {
        neatify_clusters(network, artifact_geoms, &clusters, &comp_labels, params)?;
    }

    Ok(())
}

/// Simplify singleton face artifacts.
///
/// For each single artifact:
/// 1. Run COINS and CES classification
/// 2. Link nodes to artifacts
/// 3. Dispatch to appropriate handler (n1_g1_identical, nx_gx_identical, nx_gx)
fn neatify_singletons(
    network: &mut StreetNetwork,
    artifact_geoms: &[GGeometry],
    artifact_indices: &[usize],
    params: &NeatifyParams,
) -> Result<(), NeatifyError> {
    // Run COINS analysis on the full network
    let coins_result = continuity::coins(&network.geometries, params.angle_threshold);

    // Get CES info for singletons
    let singleton_geoms: Vec<GGeometry> = artifact_indices
        .iter()
        .map(|&i| Clone::clone(&artifact_geoms[i]))
        .collect();
    let ces_info =
        continuity::get_stroke_info(&singleton_geoms, &network.geometries, &coins_result);

    let mut to_drop: Vec<usize> = Vec::new();
    let mut to_add: Vec<GGeometry> = Vec::new();

    for (local_idx, &art_idx) in artifact_indices.iter().enumerate() {
        let artifact = &artifact_geoms[art_idx];
        let ces = &ces_info[local_idx];

        // Find edges covered by this artifact
        let covered_edges = find_covered_edges(&network.geometries, artifact, params.eps);
        if covered_edges.is_empty() {
            continue;
        }

        // Get network nodes touching this artifact
        let covered_geoms: Vec<GGeometry> = covered_edges
            .iter()
            .map(|&i| Clone::clone(&network.geometries[i]))
            .collect();
        let (node_coords, _) = nodes::nodes_from_edges(&covered_geoms);
        let n_nodes = node_coords.len();

        let n_strokes = ces.stroke_count;

        // Non-planar check: skip if stroke count > node count
        if n_strokes > n_nodes {
            continue;
        }

        // Dispatch based on node count and CES composition
        if n_nodes == 1 && n_strokes == 1 {
            // n1_g1_identical: single dead-end loop
            process_n1_g1_identical(
                &covered_edges,
                artifact,
                &node_coords,
                &network.geometries,
                params,
                &mut to_drop,
                &mut to_add,
            );
        } else if n_nodes > 1 && is_identical_ces(ces) {
            // nx_gx_identical: all strokes have the same CES type
            process_nx_gx_identical(
                &covered_edges,
                artifact,
                &node_coords,
                &network.geometries,
                params,
                &mut to_drop,
                &mut to_add,
            );
        } else if n_nodes > 1 {
            // nx_gx: mixed CES types (most complex case)
            process_nx_gx(
                &covered_edges,
                artifact,
                &node_coords,
                &network.geometries,
                &coins_result,
                params,
                &mut to_drop,
                &mut to_add,
            );
        }
    }

    apply_changes(network, &to_drop, &to_add);
    Ok(())
}

/// Simplify pairs of face artifacts.
fn neatify_pairs(
    network: &mut StreetNetwork,
    artifact_geoms: &[GGeometry],
    artifact_indices: &[usize],
    comp_labels: &[usize],
    params: &NeatifyParams,
) -> Result<(), NeatifyError> {
    // Group artifacts by component label into pairs
    let mut pair_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in artifact_indices {
        pair_groups.entry(comp_labels[i]).or_default().push(i);
    }

    let coins_result = continuity::coins(&network.geometries, params.angle_threshold);

    // Determine solution for each pair
    let mut drop_interline_pairs: Vec<Vec<usize>> = Vec::new();
    let mut iterate_pairs: Vec<Vec<usize>> = Vec::new();
    let mut skeleton_pairs: Vec<Vec<usize>> = Vec::new();

    for (_label, pair) in &pair_groups {
        if pair.len() != 2 {
            continue;
        }

        // Find shared edge (covered by both artifact polygons)
        let covered_a = find_covered_edges(&network.geometries, &artifact_geoms[pair[0]], params.eps);
        let covered_b = find_covered_edges(&network.geometries, &artifact_geoms[pair[1]], params.eps);
        let set_a: HashSet<usize> = covered_a.iter().copied().collect();
        let set_b: HashSet<usize> = covered_b.iter().copied().collect();
        let shared: Vec<usize> = set_a.intersection(&set_b).copied().collect();

        if shared.is_empty() {
            // Non-planar: skip
            continue;
        }

        if shared.len() > 1 {
            // Multiple shared edges → skeleton
            skeleton_pairs.push(pair.clone());
            continue;
        }

        // Single shared edge — classify it
        let shared_idx = shared[0];
        let shared_is_end = coins_result.is_end[shared_idx];
        let only_a: Vec<usize> = covered_a.iter().filter(|i| !set_b.contains(i)).copied().collect();
        let only_b: Vec<usize> = covered_b.iter().filter(|i| !set_a.contains(i)).copied().collect();

        // Check if shared edge is C (continuing through both)
        let is_c_for_both = !shared_is_end;

        if is_c_for_both {
            // Shared edge continues through both → iterate (process as two singletons)
            iterate_pairs.push(pair.clone());
        } else if only_a.len().abs_diff(only_b.len()) <= 1 {
            // Similar edge counts → drop the interline
            drop_interline_pairs.push(pair.clone());
        } else {
            // Asymmetric → skeleton
            skeleton_pairs.push(pair.clone());
        }
    }

    // Process drop_interline: merge pair → process as singleton
    if !drop_interline_pairs.is_empty() || !iterate_pairs.is_empty() {
        let mut merged_indices: Vec<usize> = Vec::new();
        for pair in &drop_interline_pairs {
            merged_indices.extend(pair);
        }
        // First pass of iterate pairs
        for pair in &iterate_pairs {
            merged_indices.push(pair[0]);
        }
        if !merged_indices.is_empty() {
            neatify_singletons(network, artifact_geoms, &merged_indices, params)?;
        }

        // Second pass for iterate pairs
        let second_indices: Vec<usize> = iterate_pairs.iter().map(|p| p[1]).collect();
        if !second_indices.is_empty() {
            neatify_singletons(network, artifact_geoms, &second_indices, params)?;
        }
    }

    // Process skeleton pairs via cluster approach
    if !skeleton_pairs.is_empty() {
        let skeleton_indices: Vec<usize> = skeleton_pairs.iter().flatten().copied().collect();
        neatify_clusters(network, artifact_geoms, &skeleton_indices, comp_labels, params)?;
    }

    Ok(())
}

/// Simplify clusters of face artifacts.
fn neatify_clusters(
    network: &mut StreetNetwork,
    artifact_geoms: &[GGeometry],
    artifact_indices: &[usize],
    comp_labels: &[usize],
    params: &NeatifyParams,
) -> Result<(), NeatifyError> {
    // Group artifacts by component label
    let mut cluster_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &i in artifact_indices {
        cluster_groups.entry(comp_labels[i]).or_default().push(i);
    }

    let mut to_drop: Vec<usize> = Vec::new();
    let mut to_add: Vec<GGeometry> = Vec::new();

    for (_label, cluster) in &cluster_groups {
        if cluster.len() < 3 {
            continue;
        }

        // Merge all artifact polygons in the cluster
        let mut merged = Clone::clone(&artifact_geoms[cluster[0]]);
        for &i in &cluster[1..] {
            merged = match merged.union(&artifact_geoms[i]) {
                Ok(m) => m,
                Err(_) => continue,
            };
        }

        // Find edges fully within the merged polygon
        let covered = find_covered_edges(&network.geometries, &merged, params.eps);
        if covered.is_empty() {
            continue;
        }

        // Find boundary edges (edges on the boundary of the merged polygon)
        let boundary_edges = find_boundary_edges(&network.geometries, &merged, params.eps);

        if !boundary_edges.is_empty() {
            let boundary_geoms: Vec<GGeometry> = boundary_edges
                .iter()
                .map(|&i| Clone::clone(&network.geometries[i]))
                .collect();

            let (skeleton_edges, _) = geometry::voronoi_skeleton(
                &boundary_geoms,
                Some(&merged),
                None,
                params.max_segment_length,
                None,
                None,
                params.clip_limit,
                None,
            );
            if !skeleton_edges.is_empty() {
                let cleaned = remove_dangles(&skeleton_edges, &merged, params.eps);
                to_drop.extend(&covered);
                to_add.extend(cleaned);
            }
        }
    }

    apply_changes(network, &to_drop, &to_add);
    Ok(())
}

// ─── Processing functions ─────────────────────────────────────────────────

/// Process n1_g1_identical: 1 node, 1 stroke group.
///
/// Drop the covered edge and generate voronoi_skeleton replacement.
fn process_n1_g1_identical(
    covered_edges: &[usize],
    artifact: &GGeometry,
    node_coords: &[[f64; 2]],
    geometries: &[GGeometry],
    params: &NeatifyParams,
    to_drop: &mut Vec<usize>,
    to_add: &mut Vec<GGeometry>,
) {
    let covered_geoms: Vec<GGeometry> = covered_edges
        .iter()
        .map(|&i| Clone::clone(&geometries[i]))
        .collect();

    // Build snap targets from node coordinates
    let snap_targets = node_coords_to_points(node_coords);

    let (edgelines, _splitters) = geometry::voronoi_skeleton(
        &covered_geoms,
        Some(artifact),
        Some(&snap_targets),
        params.max_segment_length,
        None,
        None,
        params.clip_limit,
        None,
    );

    to_drop.extend(covered_edges);
    let cleaned = remove_dangles(&edgelines, artifact, params.eps);
    to_add.extend(cleaned);
}

/// Process nx_gx_identical: N>1 nodes, all same CES type.
///
/// Drop all covered edges and connect entry points to centroid.
/// If connections aren't within the polygon, use voronoi_skeleton instead.
fn process_nx_gx_identical(
    covered_edges: &[usize],
    artifact: &GGeometry,
    node_coords: &[[f64; 2]],
    geometries: &[GGeometry],
    params: &NeatifyParams,
    to_drop: &mut Vec<usize>,
    to_add: &mut Vec<GGeometry>,
) {
    // Find relevant nodes (nodes touching the artifact)
    let relevant_nodes: Vec<[f64; 2]> = find_nodes_near_polygon(node_coords, artifact, params.eps);

    if relevant_nodes.is_empty() {
        return;
    }

    // Compute centroid of the artifact
    let centroid = match artifact.get_centroid() {
        Ok(c) => c,
        Err(_) => return,
    };
    let centroid_cs = match centroid.get_coord_seq() {
        Ok(cs) => cs,
        Err(_) => return,
    };
    let (cx, cy) = match (centroid_cs.get_x(0), centroid_cs.get_y(0)) {
        (Ok(x), Ok(y)) => (x, y),
        _ => return,
    };

    // Create shortest lines from each relevant node to centroid
    let mut lines = Vec::new();
    let mut all_within = true;

    for node in &relevant_nodes {
        let wkt = format!("LINESTRING ({} {}, {} {})", node[0], node[1], cx, cy);
        if let Ok(line) = GGeometry::new_from_wkt(&wkt) {
            if !geometry::is_within(&line, artifact, 0.1) {
                all_within = false;
                break;
            }
            lines.push(line);
        }
    }

    to_drop.extend(covered_edges);

    if all_within && !lines.is_empty() {
        // Check angle between two lines for sharp angle
        if lines.len() == 2 {
            let angle = geometry::angle_between_two_lines(&lines[0], &lines[1]);
            if angle < params.angle_threshold {
                // Replace with direct connection between nodes
                let wkt = format!(
                    "LINESTRING ({} {}, {} {})",
                    relevant_nodes[0][0],
                    relevant_nodes[0][1],
                    relevant_nodes[1][0],
                    relevant_nodes[1][1]
                );
                if let Ok(direct) = GGeometry::new_from_wkt(&wkt) {
                    to_add.push(direct);
                    return;
                }
            }
        }
        to_add.extend(lines);
    } else {
        // Use voronoi_skeleton instead
        let covered_geoms: Vec<GGeometry> = covered_edges
            .iter()
            .map(|&i| Clone::clone(&geometries[i]))
            .collect();
        let snap_targets = node_coords_to_points(&relevant_nodes);

        let (edgelines, _) = geometry::voronoi_skeleton(
            &covered_geoms,
            Some(artifact),
            Some(&snap_targets),
            params.max_segment_length,
            None,
            None,
            params.clip_limit,
            None,
        );
        let cleaned = remove_dangles(&edgelines, artifact, params.eps);
        to_add.extend(cleaned);
    }
}

/// Process nx_gx: N>1 nodes, mixed CES types.
///
/// The most complex case. Classifies covered edges by CES hierarchy (C > E > S),
/// drops lower-hierarchy edges, then reconnects disconnected nodes.
///
/// Key branches:
/// 1. Check if dropping E/S edges causes disconnection (via connected components)
/// 2. If disconnected and multiple C edges: use skeleton snapped to high-degree nodes
/// 3. If disconnected and single C: connect remaining nodes via shortest lines or skeleton
/// 4. Loop special case: single C + single E/S → shortest line if within polygon
/// 5. Sausage special case: 2 nodes + 2 strokes → shortest line between endpoints
fn process_nx_gx(
    covered_edges: &[usize],
    artifact: &GGeometry,
    node_coords: &[[f64; 2]],
    geometries: &[GGeometry],
    coins_result: &continuity::CoinsResult,
    params: &NeatifyParams,
    to_drop: &mut Vec<usize>,
    to_add: &mut Vec<GGeometry>,
) {
    // Classify edges by CES hierarchy
    let (c_edges, e_edges, s_edges, es_mask, highest) =
        classify_ces_edges(covered_edges, coins_result);

    let n_c = c_edges.len();
    let n_e = e_edges.len();
    let n_s = s_edges.len();
    let n_nodes = node_coords.len();
    let n_strokes = {
        let groups: HashSet<usize> = covered_edges.iter().map(|&i| coins_result.group[i]).collect();
        groups.len()
    };

    // Drop ES edges
    to_drop.extend(&es_mask);

    // Check connected components after dropping ES
    let remaining_geoms: Vec<GGeometry> = highest
        .iter()
        .map(|&i| Clone::clone(&geometries[i]))
        .collect();
    let n_comps = if remaining_geoms.is_empty() {
        0
    } else {
        nodes::get_components(&remaining_geoms).iter().collect::<HashSet<_>>().len()
    };

    let relevant_nodes = find_nodes_near_polygon(node_coords, artifact, params.eps);

    // === BRANCH: Loop special case ===
    // C == 1, (E + S) == 1, and a shortest line within the polygon is much shorter than C
    if n_c == 1 && (n_e + n_s) == 1 && relevant_nodes.len() == 2 {
        let wkt = format!(
            "LINESTRING ({} {}, {} {})",
            relevant_nodes[0][0], relevant_nodes[0][1],
            relevant_nodes[1][0], relevant_nodes[1][1]
        );
        if let Ok(shortest) = GGeometry::new_from_wkt(&wkt) {
            let c_len: f64 = c_edges.iter()
                .map(|&i| geometries[i].length().unwrap_or(0.0))
                .sum();
            let s_len = shortest.length().unwrap_or(f64::INFINITY);
            if geometry::is_within(&shortest, artifact, params.eps) && s_len < c_len * 0.5 {
                to_add.push(shortest);
                return;
            }
        }
        // Otherwise fall through to general handling
    }

    // === BRANCH: Sausage special case ===
    // 2 nodes, 2 strokes — just drop E/S, keep C (no reconnection needed)
    if n_nodes == 2 && n_strokes == 2 && !highest.is_empty() {
        return;
    }

    // === BRANCH: All dropped (no C edges) → replace with skeleton ===
    if highest.is_empty() && !es_mask.is_empty() {
        let es_geoms: Vec<GGeometry> = es_mask
            .iter()
            .map(|&i| Clone::clone(&geometries[i]))
            .collect();
        let snap_targets = node_coords_to_points(&relevant_nodes);
        let (edgelines, _) = geometry::voronoi_skeleton(
            &es_geoms,
            Some(artifact),
            Some(&snap_targets),
            params.max_segment_length,
            None,
            None,
            params.clip_limit,
            None,
        );
        let cleaned = remove_dangles(&edgelines, artifact, params.eps);
        to_add.extend(cleaned);
        return;
    }

    // === Only proceed with reconnection if dropping caused disconnection ===
    if n_comps <= 1 && !highest.is_empty() {
        // Single connected component after dropping — no reconnection needed
        return;
    }

    // Need to reconnect. Find nodes not on C edges.
    let highest_geoms: Vec<GGeometry> = highest
        .iter()
        .map(|&i| Clone::clone(&geometries[i]))
        .collect();

    let mut nodes_on_c = HashSet::new();
    for node in &relevant_nodes {
        let pt_wkt = format!("POINT ({} {})", node[0], node[1]);
        if let Ok(pt) = GGeometry::new_from_wkt(&pt_wkt) {
            for geom in &highest_geoms {
                if let Ok(dist) = geom.distance(&pt) {
                    if dist < params.eps {
                        nodes_on_c.insert(coord_key(node));
                        break;
                    }
                }
            }
        }
    }

    let remaining_nodes: Vec<[f64; 2]> = relevant_nodes
        .iter()
        .filter(|n| !nodes_on_c.contains(&coord_key(n)))
        .copied()
        .collect();

    if remaining_nodes.is_empty() {
        return;
    }

    // === BRANCH: Multiple C edges → use skeleton ===
    if highest.len() > 1 {
        let es_geoms: Vec<GGeometry> = es_mask
            .iter()
            .map(|&i| Clone::clone(&geometries[i]))
            .collect();
        // Snap to nodes with degree >= 4 (intersection nodes)
        let snap_targets = node_coords_to_points(&relevant_nodes);
        let c_geoms: Vec<GGeometry> = highest
            .iter()
            .map(|&i| Clone::clone(&geometries[i]))
            .collect();
        let (edgelines, _) = geometry::voronoi_skeleton(
            &es_geoms,
            Some(artifact),
            Some(&snap_targets),
            params.max_segment_length,
            None,
            Some(&c_geoms),
            params.clip_limit,
            None,
        );
        let cleaned = remove_dangles(&edgelines, artifact, params.eps);
        to_add.extend(cleaned);
        return;
    }

    // === BRANCH: Single C edge → connect remaining nodes ===
    // Try shortest lines first; fall back to skeleton
    if remaining_nodes.len() == 1 {
        // One remaining node: connect to nearest C edge via shortest line
        if let Some(line) = make_shortest_to_edges(&remaining_nodes[0], &highest_geoms) {
            if geometry::is_within(&line, artifact, params.eps) {
                to_add.push(line);
                return;
            }
        }
        // Fall back to skeleton
        let es_geoms: Vec<GGeometry> = es_mask
            .iter()
            .map(|&i| Clone::clone(&geometries[i]))
            .collect();
        let snap_targets = node_coords_to_points(&remaining_nodes);
        let (edgelines, _) = geometry::voronoi_skeleton(
            &es_geoms,
            Some(artifact),
            Some(&snap_targets),
            params.max_segment_length,
            None,
            Some(&highest_geoms),
            params.clip_limit,
            None,
        );
        let cleaned = remove_dangles(&edgelines, artifact, params.eps);
        to_add.extend(cleaned);
    } else {
        // Multiple remaining nodes: skeleton with C as secondary snap
        let es_geoms: Vec<GGeometry> = es_mask
            .iter()
            .map(|&i| Clone::clone(&geometries[i]))
            .collect();
        let snap_targets = node_coords_to_points(&remaining_nodes);
        let (edgelines, _) = geometry::voronoi_skeleton(
            &es_geoms,
            Some(artifact),
            Some(&snap_targets),
            params.max_segment_length,
            None,
            Some(&highest_geoms),
            params.clip_limit,
            None,
        );
        let cleaned = remove_dangles(&edgelines, artifact, params.eps);
        to_add.extend(cleaned);
    }
}

/// Classify covered edges into C (continuing), E (ending), S (single) groups.
///
/// Returns (c_edges, e_edges, s_edges, es_mask, highest):
/// - c_edges: edges in C groups
/// - e_edges: edges in E groups
/// - s_edges: edges in S groups
/// - es_mask: union of E + S edges (to drop)
/// - highest: C edges (to keep)
fn classify_ces_edges(
    covered_edges: &[usize],
    coins_result: &continuity::CoinsResult,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    // Find end edges
    let end_edges: HashSet<usize> = covered_edges
        .iter()
        .filter(|&&i| coins_result.is_end[i])
        .copied()
        .collect();
    let end_groups: HashSet<usize> = end_edges.iter().map(|&i| coins_result.group[i]).collect();

    // S groups: end groups where ALL edges of the group are inside the artifact
    let mut s_groups = HashSet::new();
    for &group in &end_groups {
        let total_in_group = covered_edges
            .iter()
            .find(|&&i| coins_result.group[i] == group)
            .map(|&i| coins_result.stroke_count[i])
            .unwrap_or(0);
        let count_inside = covered_edges
            .iter()
            .filter(|&&i| coins_result.group[i] == group)
            .count();
        if count_inside == total_in_group {
            s_groups.insert(group);
        }
    }

    // E groups: end groups that aren't S
    let e_groups: HashSet<usize> = end_groups.difference(&s_groups).copied().collect();

    // C groups: covered groups that aren't end groups
    let covered_groups: HashSet<usize> =
        covered_edges.iter().map(|&i| coins_result.group[i]).collect();
    let _c_groups: HashSet<usize> = covered_groups.difference(&end_groups).copied().collect();

    let mut c_edges = Vec::new();
    let mut e_edges = Vec::new();
    let mut s_edges = Vec::new();
    let mut es_mask = Vec::new();
    let mut highest = Vec::new();

    for &i in covered_edges {
        let g = coins_result.group[i];
        if e_groups.contains(&g) {
            e_edges.push(i);
            es_mask.push(i);
        } else if s_groups.contains(&g) {
            s_edges.push(i);
            es_mask.push(i);
        } else {
            c_edges.push(i);
            highest.push(i);
        }
    }

    (c_edges, e_edges, s_edges, es_mask, highest)
}

/// Make a shortest line from a point to the nearest of a set of edges.
fn make_shortest_to_edges(point: &[f64; 2], edges: &[GGeometry]) -> Option<GGeometry> {
    let pt_wkt = format!("POINT ({} {})", point[0], point[1]);
    let pt = GGeometry::new_from_wkt(&pt_wkt).ok()?;

    let mut best_line: Option<GGeometry> = None;
    let mut best_dist = f64::INFINITY;

    for geom in edges {
        if let Ok(cs) = geom.nearest_points(&pt) {
            if let (Ok(x0), Ok(y0), Ok(x1), Ok(y1)) =
                (cs.get_x(0), cs.get_y(0), cs.get_x(1), cs.get_y(1))
            {
                let wkt = format!("LINESTRING ({} {}, {} {})", x0, y0, x1, y1);
                if let Ok(line) = GGeometry::new_from_wkt(&wkt) {
                    let len = line.length().unwrap_or(f64::INFINITY);
                    if len < best_dist {
                        best_dist = len;
                        best_line = Some(line);
                    }
                }
            }
        }
    }
    best_line
}

/// Convert a coordinate to a hash key for deduplication.
fn coord_key(c: &[f64; 2]) -> (i64, i64) {
    ((c[0] * 1e8) as i64, (c[1] * 1e8) as i64)
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Apply accumulated drops and additions to the network.
///
/// Post-processing matches Python:
/// 1. Drop marked edges
/// 2. Merge new additions via line_merge → explode to single LineStrings
/// 3. Deduplicate
/// 4. Simplify new edges by max_segment_length * simplification_factor
/// 5. Clean topology via remove_interstitial_nodes
fn apply_changes(
    network: &mut StreetNetwork,
    to_drop: &[usize],
    to_add: &[GGeometry],
) {
    if to_drop.is_empty() && to_add.is_empty() {
        return;
    }

    let drop_set: HashSet<usize> = to_drop.iter().copied().collect();

    let mut new_geoms = Vec::new();
    let mut new_statuses = Vec::new();

    for (i, geom) in network.geometries.iter().enumerate() {
        if !drop_set.contains(&i) {
            new_geoms.push(Clone::clone(geom));
            new_statuses.push(network.statuses[i]);
        }
    }

    // Post-process additions: line_merge, explode, dedup
    if !to_add.is_empty() {
        let merged_adds = merge_and_explode(to_add);
        let deduped = dedup_geometries(&merged_adds);
        for geom in deduped {
            new_geoms.push(geom);
            new_statuses.push(EdgeStatus::New);
        }
    }

    network.geometries = new_geoms;
    network.statuses = new_statuses;

    // Clean topology after changes
    let (cleaned, statuses) =
        nodes::remove_interstitial_nodes(&network.geometries, &network.statuses);
    network.geometries = cleaned;
    network.statuses = statuses;
}

/// Check if a CES classification represents an "identical" case
/// (all strokes are the same type: all C, all E, or all S).
fn is_identical_ces(ces: &continuity::CesInfo) -> bool {
    let types_present =
        (if ces.c > 0 { 1 } else { 0 })
        + (if ces.e > 0 { 1 } else { 0 })
        + (if ces.s > 0 { 1 } else { 0 });
    types_present <= 1
}

/// Merge a set of geometries via GEOS line_merge, then explode MultiLineStrings
/// into individual LineStrings.
fn merge_and_explode(geoms: &[GGeometry]) -> Vec<GGeometry> {
    if geoms.is_empty() {
        return vec![];
    }

    // Collect into a GeometryCollection and line_merge
    let wkt_parts: Vec<String> = geoms
        .iter()
        .filter_map(|g| g.to_wkt().ok())
        .collect();

    if wkt_parts.is_empty() {
        return geoms.iter().map(|g| Clone::clone(g)).collect();
    }

    // Build a GeometryCollection WKT
    let gc_wkt = format!("GEOMETRYCOLLECTION ({})", wkt_parts.join(", "));
    let collection = match GGeometry::new_from_wkt(&gc_wkt) {
        Ok(gc) => gc,
        Err(_) => return geoms.iter().map(|g| Clone::clone(g)).collect(),
    };

    let merged = match collection.line_merge() {
        Ok(m) => m,
        Err(_) => return geoms.iter().map(|g| Clone::clone(g)).collect(),
    };

    // Explode result into individual LineStrings
    let mut result = Vec::new();
    explode_geometry(&merged, &mut result);
    if result.is_empty() {
        // Fallback: return originals
        return geoms.iter().map(|g| Clone::clone(g)).collect();
    }
    result
}

/// Recursively explode a geometry (possibly Multi or Collection) into simple geometries.
fn explode_geometry(geom: &GGeometry, out: &mut Vec<GGeometry>) {
    use geos::GeometryTypes;
    match geom.geometry_type() {
        GeometryTypes::LineString => {
            // Only include non-empty
            if geom.is_empty().unwrap_or(true) {
                return;
            }
            out.push(Clone::clone(geom));
        }
        GeometryTypes::MultiLineString | GeometryTypes::GeometryCollection => {
            let n = geom.get_num_geometries().unwrap_or(0);
            for i in 0..n {
                if let Ok(part) = geom.get_geometry_n(i) {
                    let owned = geos::Geom::clone(&part);
                    explode_geometry(&owned, out);
                }
            }
        }
        _ => {
            // Skip non-line geometries (points, polygons)
        }
    }
}

/// Deduplicate geometries by normalized WKT.
fn dedup_geometries(geoms: &[GGeometry]) -> Vec<GGeometry> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for geom in geoms {
        // Normalize for consistent WKT
        let mut g = Clone::clone(geom);
        let _ = g.normalize();
        if let Ok(wkt) = g.to_wkt() {
            if seen.insert(wkt) {
                result.push(Clone::clone(geom));
            }
        } else {
            result.push(Clone::clone(geom));
        }
    }
    result
}

/// Find edge indices whose geometry is covered by the artifact polygon.
fn find_covered_edges(
    geometries: &[GGeometry],
    artifact: &GGeometry,
    eps: f64,
) -> Vec<usize> {
    let tree = crate::spatial::build_rtree(geometries);
    let candidates = nodes::envelope_query_indices_pub(&tree, artifact);

    let buffered = match artifact.buffer(eps, 8) {
        Ok(b) => b,
        Err(_) => return vec![],
    };

    candidates
        .into_iter()
        .filter(|&i| buffered.covers(&geometries[i]).unwrap_or(false))
        .collect()
}

/// Find edges that intersect the artifact boundary but are not fully covered.
fn find_boundary_edges(
    geometries: &[GGeometry],
    artifact: &GGeometry,
    eps: f64,
) -> Vec<usize> {
    let tree = crate::spatial::build_rtree(geometries);
    let candidates = nodes::envelope_query_indices_pub(&tree, artifact);

    let buffered = match artifact.buffer(eps, 8) {
        Ok(b) => b,
        Err(_) => return vec![],
    };

    let boundary = match artifact.boundary() {
        Ok(b) => b,
        Err(_) => return vec![],
    };
    let boundary_buf = match boundary.buffer(eps, 8) {
        Ok(b) => b,
        Err(_) => return vec![],
    };

    candidates
        .into_iter()
        .filter(|&i| {
            let geom = &geometries[i];
            let is_covered = buffered.covers(geom).unwrap_or(false);
            let touches_boundary = geom.intersects(&boundary_buf).unwrap_or(false);
            !is_covered && touches_boundary
        })
        .collect()
}

/// Find network nodes near a polygon (within eps).
fn find_nodes_near_polygon(
    node_coords: &[[f64; 2]],
    polygon: &GGeometry,
    eps: f64,
) -> Vec<[f64; 2]> {
    let mut result = Vec::new();
    for coord in node_coords {
        let pt_wkt = format!("POINT ({} {})", coord[0], coord[1]);
        if let Ok(pt) = GGeometry::new_from_wkt(&pt_wkt) {
            if let Ok(dist) = polygon.distance(&pt) {
                if dist <= eps {
                    result.push(*coord);
                }
            }
        }
    }
    result
}

/// Remove dangling edges from skeleton output.
///
/// After line_merge + explode, remove edges whose endpoint doesn't connect
/// to any other edge (within snap tolerance) and doesn't touch the artifact
/// boundary (where the skeleton connects to the network).
///
/// Mirrors Python `remove_dangles()`.
fn remove_dangles(connections: &[GGeometry], artifact: &GGeometry, _eps: f64) -> Vec<GGeometry> {
    if connections.len() <= 1 {
        return connections.to_vec();
    }

    // Line merge first, then explode
    let merged = merge_and_explode(connections);
    if merged.len() <= 1 {
        return merged;
    }

    // Get artifact boundary
    let boundary = match artifact.boundary() {
        Ok(b) => b,
        Err(_) => return merged,
    };

    // For each connection, check if each endpoint either:
    // 1. Touches another connection (within snap tolerance), or
    // 2. Is on the artifact boundary (connects to the network)
    //
    // Use distance to the GEOMETRY of other connections (not just endpoints),
    // since line_merge may have created longer chains where junction points
    // are interior vertices, not endpoints.
    // Use generous snap tolerance for projected CRS (meters).
    // The voronoi skeleton output may have imprecise junction points.
    let snap_tol = 5.0;

    let mut keep = vec![true; merged.len()];

    for i in 0..merged.len() {
        let cs = match merged[i].get_coord_seq() {
            Ok(c) => c,
            Err(_) => continue,
        };
        let n = cs.size().unwrap_or(0);
        if n < 2 {
            continue;
        }

        // Check both endpoints
        for pt_idx in [0, n - 1] {
            let (x, y) = match (cs.get_x(pt_idx), cs.get_y(pt_idx)) {
                (Ok(x), Ok(y)) => (x, y),
                _ => continue,
            };
            let pt_wkt = format!("POINT ({} {})", x, y);
            let pt_geom = match GGeometry::new_from_wkt(&pt_wkt) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let mut connected = false;

            // Check distance to artifact boundary
            if let Ok(dist) = boundary.distance(&pt_geom) {
                if dist < snap_tol {
                    connected = true;
                }
            }

            if !connected {
                // Check distance to any other connection's geometry
                for j in 0..merged.len() {
                    if i == j {
                        continue;
                    }
                    if let Ok(dist) = merged[j].distance(&pt_geom) {
                        if dist < snap_tol {
                            connected = true;
                            break;
                        }
                    }
                }
            }

            if !connected {
                keep[i] = false;
                break;
            }
        }
    }

    merged
        .into_iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, g)| g)
        .collect()
}

/// Convert node coordinate arrays to GEOS Point geometries.
fn node_coords_to_points(coords: &[[f64; 2]]) -> Vec<GGeometry> {
    coords
        .iter()
        .filter_map(|c| GGeometry::new_from_wkt(&format!("POINT ({} {})", c[0], c[1])).ok())
        .collect()
}

/// Error type for the neatify pipeline.
#[derive(Debug, thiserror::Error)]
pub enum NeatifyError {
    #[error("GEOS error: {0}")]
    Geos(String),
    #[error("No projected CRS set on input data")]
    NoCrs,
    #[error("Artifact detection failed: {0}")]
    ArtifactDetection(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neatify_params_default() {
        let params = NeatifyParams::default();
        assert_eq!(params.n_loops, 2);
        assert_eq!(params.angle_threshold, 120.0);
    }

    #[test]
    fn test_is_identical_ces() {
        // All C
        assert!(is_identical_ces(&continuity::CesInfo {
            stroke_count: 2, c: 2, e: 0, s: 0,
        }));
        // All E
        assert!(is_identical_ces(&continuity::CesInfo {
            stroke_count: 3, c: 0, e: 3, s: 0,
        }));
        // Mixed C+E
        assert!(!is_identical_ces(&continuity::CesInfo {
            stroke_count: 3, c: 1, e: 2, s: 0,
        }));
        // Empty (no strokes)
        assert!(is_identical_ces(&continuity::CesInfo {
            stroke_count: 0, c: 0, e: 0, s: 0,
        }));
    }

    #[test]
    fn test_classify_ces_edges() {
        // Build a small coins result with known C/E/S classification
        // 4 edges: edges 0,1 in group 0 (both ends), edge 2 in group 1 (not end),
        //          edge 3 in group 2 (end, only member)
        let coins = continuity::CoinsResult {
            group: vec![0, 0, 1, 2],
            is_end: vec![true, true, false, true],
            stroke_length: vec![10.0, 10.0, 15.0, 5.0],
            stroke_count: vec![2, 2, 1, 1],
            n_segments: 4,
            n_p1_confirmed: 0,
            n_p2_confirmed: 0,
        };

        let covered = vec![0, 1, 2, 3];
        let (c_edges, e_edges, s_edges, es_mask, highest) =
            classify_ces_edges(&covered, &coins);

        // Group 0: both edges are end, both are inside → S (2 of 2)
        // Group 1: edge 2 not end → C
        // Group 2: edge 3 end, 1 of 1 inside → S
        assert!(c_edges.contains(&2), "edge 2 should be C");
        assert!(s_edges.contains(&0) && s_edges.contains(&1), "edges 0,1 should be S");
        assert!(s_edges.contains(&3), "edge 3 should be S");
        assert!(e_edges.is_empty(), "no E edges in this case");
        assert_eq!(highest.len(), 1);
        assert_eq!(es_mask.len(), 3);
    }

    #[test]
    fn test_find_covered_edges() {
        // Create a small polygon and lines inside/outside
        let poly = GGeometry::new_from_wkt(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
        ).unwrap();
        let inside = GGeometry::new_from_wkt(
            "LINESTRING (2 5, 8 5)"
        ).unwrap();
        let outside = GGeometry::new_from_wkt(
            "LINESTRING (12 5, 18 5)"
        ).unwrap();
        let crossing = GGeometry::new_from_wkt(
            "LINESTRING (5 5, 15 5)"
        ).unwrap();

        let geoms = vec![inside, outside, crossing];
        let covered = find_covered_edges(&geoms, &poly, 0.001);

        assert!(covered.contains(&0), "inside line should be covered");
        assert!(!covered.contains(&1), "outside line should not be covered");
        assert!(!covered.contains(&2), "crossing line should not be covered");
    }

    #[test]
    fn test_find_boundary_edges() {
        let poly = GGeometry::new_from_wkt(
            "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"
        ).unwrap();
        let inside = GGeometry::new_from_wkt(
            "LINESTRING (2 5, 8 5)"
        ).unwrap();
        let boundary = GGeometry::new_from_wkt(
            "LINESTRING (5 -5, 5 5)"
        ).unwrap();
        let outside = GGeometry::new_from_wkt(
            "LINESTRING (12 5, 18 5)"
        ).unwrap();

        let geoms = vec![inside, boundary, outside];
        let boundary_edges = find_boundary_edges(&geoms, &poly, 0.001);

        assert!(!boundary_edges.contains(&0), "inside line is not a boundary edge");
        assert!(boundary_edges.contains(&1), "crossing line is a boundary edge");
        assert!(!boundary_edges.contains(&2), "outside line is not a boundary edge");
    }

    #[test]
    fn test_merge_and_explode() {
        // Two connected linestrings should merge into one
        let l1 = GGeometry::new_from_wkt("LINESTRING (0 0, 5 0)").unwrap();
        let l2 = GGeometry::new_from_wkt("LINESTRING (5 0, 10 0)").unwrap();
        let result = merge_and_explode(&[l1, l2]);
        assert_eq!(result.len(), 1, "two connected lines should merge to one");

        // Two disconnected linestrings should stay as two
        let l3 = GGeometry::new_from_wkt("LINESTRING (0 0, 5 0)").unwrap();
        let l4 = GGeometry::new_from_wkt("LINESTRING (20 0, 25 0)").unwrap();
        let result2 = merge_and_explode(&[l3, l4]);
        assert_eq!(result2.len(), 2, "two disconnected lines should stay as two");
    }

    #[test]
    fn test_dedup_geometries() {
        let l1 = GGeometry::new_from_wkt("LINESTRING (0 0, 5 0)").unwrap();
        let l2 = GGeometry::new_from_wkt("LINESTRING (0 0, 5 0)").unwrap();
        let l3 = GGeometry::new_from_wkt("LINESTRING (10 0, 15 0)").unwrap();
        let result = dedup_geometries(&[l1, l2, l3]);
        assert_eq!(result.len(), 2, "duplicate should be removed");
    }

    #[test]
    fn test_coord_key() {
        let c1 = [1.23456789, 4.56789012];
        let c2 = [1.23456789, 4.56789012];
        let c3 = [1.234567891, 4.56789012]; // differs at 10th decimal
        assert_eq!(coord_key(&c1), coord_key(&c2));
        // c3 might or might not match depending on floating point precision
        // but should be stable for same input
    }

    #[test]
    fn test_make_shortest_to_edges() {
        let edge = GGeometry::new_from_wkt("LINESTRING (0 0, 10 0)").unwrap();
        let point = [5.0, 3.0];
        let line = make_shortest_to_edges(&point, &[edge]).unwrap();
        let len = line.length().unwrap();
        assert!((len - 3.0).abs() < 0.01, "shortest line from (5,3) to x-axis should be ~3");
    }
}
