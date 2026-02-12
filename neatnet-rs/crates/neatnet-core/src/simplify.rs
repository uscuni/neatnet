//! Main simplification pipeline: neatify and its sub-steps.
//!
//! Ports Python `neatnet.simplify`: `neatify`, `neatify_loop`,
//! `neatify_singletons`, `neatify_pairs`, `neatify_clusters`.

use geos::{Geom, Geometry as GGeometry};

use crate::artifacts;
use crate::continuity;
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

    // Step 3: Detect artifacts
    let artifacts = artifacts::detect_artifacts(
        &network.geometries,
        params.artifact_threshold,
        params.artifact_threshold_fallback,
    );

    let (artifact_geoms, artifact_fais, threshold) = match artifacts {
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
            let _re_artifacts = artifacts::detect_artifacts(
                &network.geometries,
                Some(threshold),
                params.artifact_threshold_fallback,
            );
            // If no more artifacts, stop early
            if _re_artifacts.is_none() {
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
    // 1. Remove edges fully within artifacts (dangles)
    // TODO: spatial index query for "contains" predicate

    // 2. Remove interstitial nodes after dangle removal
    let (cleaned, statuses) =
        nodes::remove_interstitial_nodes(&network.geometries, &network.statuses);
    network.geometries = cleaned;
    network.statuses = statuses;

    // 3. Build contiguity graph on artifacts → classify as singles/pairs/clusters
    let adjacency = artifacts::build_contiguity_graph(artifact_geoms, true);
    let comp_labels = artifacts::component_labels_from_adjacency(&adjacency);

    // Count component sizes
    let mut comp_sizes: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
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

    // 4. Simplify singletons
    if !singles.is_empty() {
        neatify_singletons(network, artifact_geoms, &singles, params)?;
    }

    // 5. Simplify pairs
    if !pairs.is_empty() {
        neatify_pairs(network, artifact_geoms, &pairs, &comp_labels, params)?;
    }

    // 6. Simplify clusters
    if !clusters.is_empty() {
        neatify_clusters(network, artifact_geoms, &clusters, &comp_labels, params)?;
    }

    Ok(())
}

/// Simplify singleton face artifacts.
///
/// For each single artifact:
/// 1. Extract covered edges
/// 2. Run COINS continuity analysis
/// 3. Classify with CES typology
/// 4. Dispatch to appropriate handler (n1_g1, nx_gx_identical, nx_gx)
fn neatify_singletons(
    network: &mut StreetNetwork,
    artifact_geoms: &[GGeometry],
    artifact_indices: &[usize],
    params: &NeatifyParams,
) -> Result<(), NeatifyError> {
    // Run COINS analysis on the full network
    let coins_result = continuity::coins(&network.geometries, params.angle_threshold);

    let mut to_drop: Vec<usize> = Vec::new();
    let to_add: Vec<GGeometry> = Vec::new();

    for &art_idx in artifact_indices {
        let artifact = &artifact_geoms[art_idx];

        // Get edges covered by this artifact (spatial query)
        let covered_edges = find_covered_edges(&network.geometries, artifact, params.eps);

        if covered_edges.is_empty() {
            continue;
        }

        // Get node count for this artifact
        let (node_coords, _) = nodes::nodes_from_edges(
            &covered_edges
                .iter()
                .map(|&i| &network.geometries[i])
                .collect::<Vec<_>>()
                .iter()
                .map(|g| Clone::clone(*g))
                .collect::<Vec<_>>(),
        );
        let n_nodes = node_coords.len();

        // Get COINS info for covered edges
        let edge_groups: Vec<usize> = covered_edges
            .iter()
            .map(|&i| coins_result.group[i])
            .collect();
        let n_strokes = edge_groups.iter().collect::<std::collections::HashSet<_>>().len();

        // Dispatch by typology (simplified)
        if n_nodes == 1 && n_strokes == 1 {
            // n1_g1_identical
            to_drop.extend(&covered_edges);
            // TODO: voronoi_skeleton replacement
        } else if n_nodes > 1 {
            // nx_gx or nx_gx_identical
            // TODO: full CES typology dispatch
        }
    }

    // Apply drops
    let mut keep_mask = vec![true; network.geometries.len()];
    for &idx in &to_drop {
        if idx < keep_mask.len() {
            keep_mask[idx] = false;
        }
    }

    let mut new_geoms = Vec::new();
    let mut new_statuses = Vec::new();
    for (i, geom) in network.geometries.iter().enumerate() {
        if keep_mask[i] {
            new_geoms.push(Clone::clone(geom));
            new_statuses.push(network.statuses[i]);
        }
    }

    // Add new geometries
    for geom in to_add {
        new_geoms.push(geom);
        new_statuses.push(EdgeStatus::New);
    }

    network.geometries = new_geoms;
    network.statuses = new_statuses;

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
    // TODO: Full pairs implementation
    // This involves:
    // 1. For each pair, determine solution (drop_interline/iterate/skeleton)
    // 2. Dispatch to appropriate handler
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
    // TODO: Full clusters implementation
    // This involves:
    // 1. For each cluster, merge polygons
    // 2. Find edges within merged polygon
    // 3. Skeletonize and replace
    Ok(())
}

/// Find edge indices whose geometry is covered by the artifact polygon.
fn find_covered_edges(
    geometries: &[GGeometry],
    artifact: &GGeometry,
    eps: f64,
) -> Vec<usize> {
    let mut covered = Vec::new();
    let buffered = match artifact.buffer(eps, 8) {
        Ok(b) => b,
        Err(_) => return covered,
    };

    for (i, geom) in geometries.iter().enumerate() {
        if buffered.covers(geom).unwrap_or(false) {
            covered.push(i);
        }
    }
    covered
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
}
