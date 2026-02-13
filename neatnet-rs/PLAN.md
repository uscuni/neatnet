# Plan: Finishing the neatnet Rust Port

## Current State

36 tests pass. Topology fixing is within 4 edges of Python.
The full pipeline outputs 734 edges vs Python's 527 — the 207-edge gap is
entirely from stubbed simplification stages.

### What works
- **types.rs** — StreetNetwork, EdgeStatus, NeatifyParams, Artifacts, CoinsInfo
- **spatial.rs** — R*-tree build/query via rstar (fixed Polygon envelope handling)
- **continuity.rs** — COINS algorithm (angle semantics fixed)
- **nodes.rs** — nodes_from_edges, get_components, remove_interstitial_nodes, fix_topology (dedup + induce + merge), fcluster, consolidate_nodes (full spider geometry), induce_nodes (degree mismatch + loop contacts + split)
- **artifacts.rs** — detect_artifacts (polygonize + FAI + KDE threshold), build_contiguity_graph, component_labels, shape metrics
- **gaps.rs** — close_gaps, extend_lines
- **geometry.rs** — voronoi_skeleton (Delaunay-based), segmentize, angle_between_two_lines, is_within, line_segments, snap_to_targets
- **simplify.rs** — neatify pipeline skeleton, neatify_loop classification
- **neatnet-py** — PyO3 bindings (WKT interface: neatify, coins, voronoi_skeleton, version)

### What's stubbed or broken
| Component | File | Status | Effort |
|---|---|---|---|
| neatify_singletons full dispatch | simplify.rs | Only n1_g1 stub, needs all CES typologies | L |
| neatify_pairs | simplify.rs | No-op stub | L |
| neatify_clusters | simplify.rs | No-op stub | M |
| Artifact processing functions | artifacts.rs (new) | n1_g1, nx_gx, nx_gx_identical, nx_gx_cluster | XL |
| GeoArrow FFI | neatnet-py | Currently WKT, needs zero-copy Arrow | M |

---

## Phase 1: Fix COINS angle semantics

**Priority: Critical** — everything downstream depends on correct stroke grouping.

**Bug:** The Rust COINS computes deflection angle (0° = collinear same-direction)
and checks `angle < threshold`, but Python computes interior angle (180° = collinear)
and checks `angle > threshold`. For a 90° perpendicular turn with threshold=120°:
- Rust: deflection=90° < 120° → MERGE (wrong)
- Python: interior=90° < 120° → DON'T MERGE (correct)

**Fix in `continuity.rs`:**

```rust
// Change deflection_angle() to return interior angle instead:
fn deflection_angle(...) -> f64 {
    // ...existing vector computation...
    let cos_angle = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
    cos_angle.acos()  // interior angle: 180° for collinear, 0° for opposite
}

// Change threshold comparison (line ~92):
if best_angle > threshold_rad {  // was: best_angle < threshold_rad
    best_pairs.push((si, sj, best_angle));
}

// Change best-link selection (line ~80):
if angle > best_angle {  // was: angle < best_angle
    best_angle = angle;
    best_partner = Some(j);
}
// Initialize best_angle to 0.0 instead of f64::MAX
```

**Validation:** Run compare_coins.py — should produce ~1158 strokes (matching Python).

---

## Phase 2: Complete voronoi_skeleton edge extraction

**Priority: Critical** — used by all artifact simplification functions.

The Voronoi diagram is already built via `voronoice`. The missing piece is
extracting edges between cells belonging to different input lines — the
"ridge extraction" step.

**Key insight from Python:** `scipy.spatial.Voronoi` provides:
- `ridge_points`: pairs of input point indices for each ridge
- `ridge_vertices`: vertex indices for ridge endpoints

`voronoice` doesn't directly expose ridge_points. Two options:

**Option A — Use Delaunay triangulation directly:**
The `voronoice` crate builds on `delaunator`. Extract the Delaunay
triangulation, compute dual edges (Voronoi ridges) from triangle
adjacency, and map back to input point IDs.

**Option B — Switch to `spade` for Voronoi:**
The `spade` crate provides constrained Delaunay triangulation with
direct access to Voronoi dual edges and face adjacency. This is a
better fit for the ridge extraction pattern.

**Implementation in `geometry.rs`:**
1. Replace `voronoice` with `spade` (or add Delaunay-based extraction)
2. For each Voronoi edge, get the two input points that generated it
3. Map input points → source line IDs
4. Keep edges where source line IDs differ (cross-line skeleton)
5. Drop edges from buffer boundary
6. Group edges by line-pair, line_merge each group
7. Clip to polygon (intersection with `poly.buffer(-dist)`)
8. Add snap-to-target connections via shortest_line

**Also needed:**
- `snap_to_targets()` — partially done, needs component-wise snapping
- `_remove_sliver()` — filter out degenerate slivers from merged result

---

## Phase 3: Complete consolidate_nodes ✓

**DONE.** Full spider geometry generation implemented with:
- DBSCAN-like pre-filtering using R-tree proximity graph (avoids O(n²))
- Hierarchical clustering (kodama) within each proximity component
- Cookie geometry (convex hull buffered by tolerance/2)
- Spider line generation from boundary intersection points to centroid
- MultiLineString explosion and empty geometry removal
- Final remove_interstitial_nodes cleanup
- Also fixed Polygon envelope handling bug in spatial.rs build_rtree

---

## Phase 4: Implement induce_nodes ✓

**DONE.** Full implementation with:
- `identify_degree_mismatch()`: spatial index query for expected vs observed degree
- `makes_loop_contact()`: loop vertex detection for non-loop and loop-loop contacts
- `split_edges_at_points()`: iterative edge splitting with R-tree queries
- `snap_n_split()`: GEOS snap + coordinate-based split at interior vertices
- `fix_topology()` now calls induce_nodes between dedup and remove_interstitial_nodes

---

## Phase 5: Implement get_artifacts iterative expansion ✓

**DONE.** Full iterative expansion implemented:
- `get_artifacts()` function with all expansion parameters
- `compute_enclosed_touching()` from rook contiguity graph
- Block-like, circle-like enclosed, circle-like touching expansion criteria
- Loop until convergence (no new artifacts)
- Exclusion mask filtering
- Fixed Polygon envelope handling in `build_contiguity_graph`
- `simplify.rs` updated to use `get_artifacts` instead of `detect_artifacts`

---

## Phase 6: Implement artifact processing functions ✓

**DONE.** CES classification and full artifact processing dispatch implemented.

### Implemented:
- **6a: CES typology** — `get_stroke_info()` in continuity.rs classifies strokes as C/E/S
- **6b: n1_g1_identical** — `process_n1_g1_identical()` drops edge, replaces with skeleton
- **6c: nx_gx_identical** — `process_nx_gx_identical()` centroid connection or skeleton fallback
- **6d: nx_gx** — `process_nx_gx()` with CES hierarchy, connected components check,
  loop/sausage special cases, multi-C skeleton, remaining node reconnection
- **6e: neatify_clusters** — merge polygons, find boundary, skeletonize

### Still simplified vs Python:
- `process_nx_gx` covers main branches but not all sub-branches (CCSS special case,
  filter_connections, avoid_forks, reconnect helpers)
- `n1_g1_identical` doesn't use COINS on skeleton output to select relevant stroke

### Bug fixes applied:
- **Shape metrics**: Fixed isoareal_quotient and isoperimetric_quotient to match
  esda (Altman PA_3 and PA_1 formulas). Fixed MBC ratio to use Welzl's algorithm.
- **remove_dangles**: Added post-skeleton dangle cleanup (5m snap tolerance for EPSG:3857)
- **Artifact re-detection**: Fixed loop 2 to use re-detected artifacts instead of stale originals
- **Cluster skeleton decomposition**: Boundary-segment clipping + entry-point decomposition
  for large clusters, matching Python's nx_gx_cluster approach
- **filter_connections / reconnect_c_groups**: Multi-C skeleton post-processing chain

### Current accuracy (Apalachicola):
- Rust: ~547 edges, length ~63,546 | Python: 527 edges, length 64,566
- Artifact detection: 89→92 (Python: 88→91) — near-perfect match
- Edge count ratio: 1.04 (within 4% of Python)
- Remaining gap (~20 edges): minor skeleton output differences in cluster processing

---

## Phase 7: Wire up neatify_singletons / pairs / clusters ✓

**DONE.** Full pipeline orchestration implemented.

### neatify_singletons (`simplify.rs`)
1. Run COINS on full network ✓
2. Classify with CES typology ✓
3. Non-planar detection (stroke_count > node_count → skip) ✓
4. Dispatch to n1_g1 / nx_gx_identical / nx_gx ✓
5. Post-processing: line_merge, explode, dedup ✓
6. Clean topology via remove_interstitial_nodes ✓

### neatify_pairs (`simplify.rs`) ✓
1. Group artifacts by component label into pairs ✓
2. For each pair, find shared edge ✓
3. Classify shared edge as C/E/S ✓
4. Determine solution: drop_interline / iterate / skeleton / non_planar ✓
5. Dispatch: drop_interline+iterate → neatify_singletons, skeleton → neatify_clusters ✓
6. Clean topology ✓

### neatify_clusters (`simplify.rs`) ✓
1. Group artifacts by component label into clusters (3+) ✓
2. Merge cluster polygons ✓
3. Drop interior edges ✓
4. Skeletonize boundary edges ✓
5. Clean topology ✓

---

## Phase 8: GeoArrow FFI for zero-copy Python interface ✓

**DONE.** GeoArrow FFI implemented:
- `neatify()` accepts `PyGeoArray` (GeoArrow LineString), returns `(PyGeoArray, PyArrow StringArray)`
- `coins()` accepts `PyGeoArray`, returns dict
- Conversion: GeoArrow → geo_types → geos (pipeline) → geo_types → GeoArrow
- Status column as `pyarrow.StringArray` via Arrow PyCapsule FFI
- WKT functions kept as `neatify_wkt/coins_wkt/voronoi_skeleton_wkt` for compatibility
- Dependencies: `geos` with `"geo"` feature, `geo-traits` for scalar conversion
- Python callers use `gdf.geometry.to_arrow(geometry_encoding="geoarrow")`
- On Apalachicola (1782 edges): WKT ~0.9s, GeoArrow ~0.95s (parity on small data)

---

## Phase 9: Testing and validation

1. **Unit tests per function** — test each artifact processing function
   against known Python output (save reference WKT from Python runs)
2. **Integration test on Apalachicola** — must match Python within tolerance:
   - 527 edges, total length 64,566, status counts matching
3. **Integration test on larger datasets** — Aleppo (78K edges), Auckland, etc.
4. **Benchmark with criterion** — target 5x speedup on Aleppo
5. **Python roundtrip test** — verify GeoDataFrame in → GeoDataFrame out

---

## Suggested implementation order

```
Phase 1: Fix COINS angle bug              [~1 hour]
Phase 2: voronoi_skeleton edge extraction  [~2 days]
Phase 3: consolidate_nodes spider geometry [~1 day]
Phase 4: induce_nodes                      [~1 day]
Phase 5: get_artifacts expansion           [~0.5 day]
Phase 6: Artifact processing functions     [~3-4 days]
  6a: CES classification                   [~0.5 day]
  6b: n1_g1_identical                      [~0.5 day]
  6c: nx_gx_identical                      [~0.5 day]
  6d: nx_gx (complex)                      [~2 days]
  6e: nx_gx_cluster                        [~0.5 day]
Phase 7: Wire up simplify pipeline         [~1 day]
Phase 8: GeoArrow FFI                      [~1 day]
Phase 9: Testing and validation            [~2 days]
```

Total: ~12-14 days of focused work.

The critical path is: Phase 1 → Phase 2 → Phase 6d → Phase 7 → Phase 9.
Phases 3-5 can be done in parallel with Phase 2.
