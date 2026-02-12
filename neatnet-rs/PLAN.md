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
| get_artifacts iterative expansion | artifacts.rs | Only initial threshold, no block/circle expansion | M |
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

## Phase 5: Implement get_artifacts iterative expansion

**Priority: High** — current version only applies the initial KDE threshold.

**Implementation in `artifacts.rs`:**

After the initial FAI threshold classification, add iterative expansion:
1. Build rook contiguity graph on all polygons
2. For each non-artifact polygon, check:
   - **Block-like:** `isoareal_quotient < 0.5` AND `area < 1e5` AND
     (enclosed by artifacts OR touching artifacts)
   - **Circle-like enclosed:** `isoareal_quotient >= 0.75` AND `area < 5e4` AND
     fully enclosed by artifacts
   - **Circle-like touching:** `isoperimetric_quotient >= 0.9` AND `area < 5e4` AND
     touching artifacts
3. Mark newly identified artifacts
4. Repeat until no new artifacts found
5. Filter by exclusion mask if provided

**Requires:** `enclosed` / `touching` relationship computation from contiguity graph.

---

## Phase 6: Implement artifact processing functions

**Priority: Critical** — these are the core simplification algorithms.

### 6a: CES typology classification (`simplify.rs`)

For each artifact polygon, classify its boundary edges as:
- **C** (Continuing): edge's COINS stroke continues through the artifact
- **E** (Ending): edge's stroke ends at the artifact
- **S** (Single): edge is the only member of its stroke group

Classification uses `_link_nodes_artifacts()` (sparse COO node-artifact
incidence) and `_classify_strokes()` (COINS group analysis).

### 6b: n1_g1_identical (1 node, 1 group)

Create new file `artifacts_processing.rs` or extend `artifacts.rs`:
1. Drop the single covered edge
2. Generate voronoi_skeleton of its segments
3. Find skeleton part intersecting the node endpoint
4. Return as replacement

### 6c: nx_gx_identical (N nodes, all same group)

1. Drop all covered edges
2. Connect each node to polygon centroid via shortest_line
3. If connection not within polygon → use voronoi_skeleton instead
4. Angle check: if two connections form angle < threshold, replace with direct line

### 6d: nx_gx (N≥2 nodes, M≥2 groups) — the complex case

This is the largest single function (~360 lines in Python). Three main branches:

**Branch 1 — Multiple C edges (highest priority):**
- Use voronoi_skeleton snapped to degree≥4 nodes
- Special CCSS case: connect via midpoints if S << C
- Filter connections to avoid forks
- Reconnect disconnected components

**Branch 2 — Relevant high-degree nodes, single C:**
- 1 remaining node → direct shortest_line or skeleton
- Multiple remaining → skeleton connecting all

**Branch 3 — No relevant nodes, snap to C continuity:**
- Similar sub-branches

Also handles: loops, sausages, non-planar edge cases.

**Helper functions needed:**
- `filter_connections()` — keep only necessary connections
- `avoid_forks()` — remove duplicate forks at endpoint
- `reconnect()` — connect disconnected C components
- `remove_dangles()` — clean dangling edges
- `one_remaining()` / `multiple_remaining()` — isolated node handlers
- `is_dangle()` — check if edge has degree-1 endpoint

### 6e: nx_gx_cluster (cluster of 2+ artifacts)

1. Merge all artifact polygons
2. Drop all edges fully within merged polygon
3. Find boundary edges
4. voronoi_skeleton(boundary_edges, merged_polygon)
5. Reconnect non-planar intruding edges

---

## Phase 7: Wire up neatify_singletons / pairs / clusters

**Priority: Critical** — orchestration connecting phases 5-6 to the pipeline.

### neatify_singletons (`simplify.rs`)
1. Link nodes to artifacts via spatial index
2. Run COINS if not cached
3. Classify with CES typology
4. Filter non-planar artifacts
5. Dispatch each artifact to n1_g1 / nx_gx_identical / nx_gx
6. Apply drops + additions + split at new nodes
7. Clean topology

### neatify_pairs (`simplify.rs`)
1. Group artifacts by component label into pairs
2. For each pair, find shared edge
3. Classify shared edge as C/E/S
4. Determine solution: drop_interline / iterate / skeleton / non_planar
5. Dispatch by solution type
6. Clean topology

### neatify_clusters (`simplify.rs`)
1. Group artifacts by component label into clusters (3+)
2. Merge cluster polygons
3. Drop interior edges
4. Skeletonize boundary edges
5. Reconnect non-planar intruders
6. Clean topology

---

## Phase 8: GeoArrow FFI for zero-copy Python interface

**Priority: Medium** — WKT works for correctness testing, GeoArrow for performance.

Replace WKT serialization in `neatnet-py/src/lib.rs` with:
1. Accept `geoarrow.ChunkedArray` via `pyo3-geoarrow`
2. Convert to `Vec<geos::Geometry>` using geoarrow's WKB or coordinate access
3. Run pipeline
4. Convert result back to `geoarrow.ChunkedArray`
5. Return Arrow arrays for status column

This eliminates the WKT parse/serialize overhead (~50% of current Rust time
on small datasets).

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
