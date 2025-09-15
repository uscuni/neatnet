import collections.abc
import typing
import warnings

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import shapely
from scipy import sparse
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN


def _fill_attrs(gdf: gpd.GeoDataFrame, source_row: pd.Series) -> gpd.GeoDataFrame:
    """Thoughtful attribute assignment to lines split into segments by new nodes –
    taking list-like values into consideration. See gh#213. Regarding iterables,
    currently only supports list values – others can be added based on input type
    in the future on an ad hoc basis as problems arise. Called from within ``split()``.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The new frame of split linestrings.
    source_row: pandas.Series
        The original source row.

    Returns
    -------
    geopandas.GeoDataFrame
        The input ``gdf`` with updated columns based on values in ``source_row``.
    """

    def _populate_column(attr):
        """Return the attribute if scalar, create vector of input if not."""
        if isinstance(attr, collections.abc.Sequence) and not isinstance(attr, str):
            attr = [attr] * gdf.shape[0]
        return attr

    for col in source_row.index.drop(["geometry", "_status"], errors="ignore"):
        gdf[col] = _populate_column(source_row[col])

    return gdf


def split(
    split_points: list | np.ndarray | gpd.GeoSeries,
    cleaned_streets: gpd.GeoDataFrame,
    crs: str | pyproj.CRS,
    *,
    eps: float = 1e-4,
) -> gpd.GeoSeries | gpd.GeoDataFrame:
    """Split lines on new nodes.

    Parameters
    ----------
    split_points : list | numpy.ndarray
        Points to split the ``cleaned_roads``.
    cleaned_streets : geopandas.GeoDataFrame
        Line geometries to be split with ``split_points``.
    crs : str | pyproj.CRS
        Anything accepted by ``pyproj.CRS``.
    eps : float = 1e-4
        Tolerance epsilon for point snapping.

    Returns
    -------
    geopandas.GeoSeries | geopandas.GeoDataFrame
        Resultant split line geometries.
    """
    split_points = gpd.GeoSeries(split_points, crs=crs)
    for split in split_points.drop_duplicates():
        _, ix = cleaned_streets.sindex.nearest(split, max_distance=eps)
        row = cleaned_streets.iloc[ix]
        edge = row.geometry
        if edge.shape[0] == 1:
            row = row.iloc[0]
            lines_split = _snap_n_split(edge.item(), split, eps)
            if lines_split.shape[0] > 1:
                gdf_split = gpd.GeoDataFrame(geometry=lines_split, crs=crs)
                gdf_split = _fill_attrs(gdf_split, row)
                gdf_split["_status"] = "changed"
                cleaned_streets = pd.concat(
                    [cleaned_streets.drop(edge.index[0]), gdf_split],
                    ignore_index=True,
                )
        elif edge.shape[0] > 1:
            to_be_dropped = []
            to_be_added = []
            for i, e in edge.items():
                lines_split = _snap_n_split(e, split, eps)
                if lines_split.shape[0] > 1:
                    to_be_dropped.append(i)
                    to_be_added.append(lines_split)

            if to_be_added:
                gdf_split = pd.DataFrame(
                    {"geometry": to_be_added, "_orig": to_be_dropped}
                ).explode("geometry")
                gdf_split = pd.concat(
                    [
                        gdf_split.drop(columns="_orig").reset_index(drop=True),
                        row.drop(columns="geometry")
                        .loc[gdf_split["_orig"]]
                        .reset_index(drop=True),
                    ],
                    axis=1,
                )
                gdf_split["_status"] = "changed"
                cleaned_streets = pd.concat(
                    [cleaned_streets.drop(to_be_dropped), gdf_split],
                    ignore_index=True,
                )
                cleaned_streets = gpd.GeoDataFrame(
                    cleaned_streets, geometry="geometry", crs=crs
                )

    return cleaned_streets.reset_index(drop=True)


def _snap_n_split(e: shapely.LineString, s: shapely.Point, tol: float) -> np.ndarray:
    """Snap point to edge and return lines to split."""
    snapped = shapely.snap(e, s, tolerance=tol)
    _lines_split = shapely.get_parts(shapely.ops.split(snapped, s))
    return _lines_split[~shapely.is_empty(_lines_split)]


def _status(x: pd.Series) -> str:
    """Determine the status of edge line(s)."""
    if len(x) == 1:
        return x.iloc[0]
    return "changed"


def isolate_bowtie_nodes(edgelines: list | np.ndarray | gpd.GeoSeries) -> gpd.GeoSeries:
    r"""
    Bowties are a rare edgecase whereby a component has:
    * 2 unique nodes
    * 2 edges that are loops
    * 4 edges that have only 3 unique coord-pairs *after* sorting.

        |\ /‾‾\ /|
        | *    * |
        |/ \__/ \|

    Although extremely rare, these cases throw a wrench in the
    efficient logic of ``get_components()``. See gh#214.
    """

    ignore = []

    mm_nx = momepy.gdf_to_nx(gpd.GeoDataFrame(geometry=edgelines))
    mm_nx_cc = nx.connected_components(mm_nx)

    potential_bowtie_cc_nodes = []
    for cc in list(mm_nx_cc):
        # potential bowties only have 2 unique nodes
        if len(cc) == 2:
            potential_bowtie_cc_nodes += [list(cc)]

    if potential_bowtie_cc_nodes:
        for potential_bowtie_cc in potential_bowtie_cc_nodes:
            comp_edges = mm_nx.subgraph(potential_bowtie_cc).edges

            # potential bowties have 2 edges that are loops
            loops = [comp_edges[ce]["geometry"].is_closed for ce in comp_edges]

            if sum(loops) == 2:
                # -- failsafe
                # ensure the 4 edges have only 3 unique coord-pairs after sorting
                sorted_edge_coords = [sorted(ce[:2]) for ce in comp_edges]
                unique_sorted_edge_coords = np.unique(sorted_edge_coords, axis=0)

                if unique_sorted_edge_coords.shape[0] == 3:
                    ignore += potential_bowtie_cc

    return gpd.GeoSeries([shapely.Point(i) for i in ignore]).sort_values(
        ignore_index=True
    )


def get_components(
    edgelines: list | np.ndarray | gpd.GeoSeries,
    *,
    ignore: None | gpd.GeoSeries = None,
) -> np.ndarray:
    """Determine groups of chained edges ("components" in this function) that are
    linked by degree-2 nodes. These edges are given component labels that are then
    used to aggregate the chained edges into single, LineString geometries.

    Parameters
    ----------
    edgelines : list | np.ndarray | gpd.GeoSeries
        Collection of line objects.
    ignore : None | gpd.GeoSeries = None
        Nodes to ignore when labeling components.

    Returns
    -------
    np.ndarray
        Component labels to aggregate chains of edges bounded by
        degree-2 nodes into single, LineString geometries.

    Notes
    -----
    See [https://github.com/uscuni/neatnet/issues/56] and
    [https://github.com/uscuni/neatnet/pull/235] for detailed explanation of output.
    """

    # 1. tease out any "bowtie" cases - don't consider those nodes to merge
    bowtie_nodes = isolate_bowtie_nodes(edgelines)
    if not bowtie_nodes.empty:
        if ignore is not None:
            ignore = pd.concat([ignore, bowtie_nodes], ignore_index=True)
        else:
            ignore = bowtie_nodes

    # 2. convert to numpy array for operations
    edgelines = np.array(edgelines)

    # 3. isolate starting and ending nodes of edges
    start_points = shapely.get_point(edgelines, 0)
    end_points = shapely.get_point(edgelines, -1)

    # 4. consider only unique nodes
    points = shapely.points(
        np.unique(
            shapely.get_coordinates(np.concatenate([start_points, end_points])), axis=0
        )
    )

    # 5. filter out any pre-defined nodes to ignore when considering merging
    if ignore is not None:
        mask = np.isin(points, ignore)
        points = points[~mask]

    # 6. query nodes that intersect non-loop edges
    inp, res = shapely.STRtree(shapely.boundary(edgelines)).query(
        points, predicate="intersects"
    )

    # 7. filter nodes that intersect exactly 2 edges
    unique, counts = np.unique(inp, return_counts=True)
    mask = np.isin(inp, unique[counts == 2])
    # node index to consider for merging
    merge_inp = inp[mask]
    # edge index to consider for merging
    merge_res = res[mask]

    # 8. generate loop index from input edgelines
    closed = np.arange(len(edgelines))[shapely.is_closed(edgelines)]

    # 9. update mask with loop edges
    mask = np.isin(merge_inp, closed) | np.isin(merge_res, closed)

    # 10. invert mask for final filter of nodes/edges to merge
    merge_res = merge_res[~mask]
    merge_inp = merge_inp[~mask]

    # 11. generate network component topology for edges to merge
    g = nx.Graph(list(zip((merge_inp * -1) - 1, merge_res, strict=True)))
    components = {
        i: {v for v in k if v > -1} for i, k in enumerate(nx.connected_components(g))
    }
    component_labels = {value: key for key in components for value in components[key]}
    labels = pd.Series(component_labels, index=range(len(edgelines)))
    max_label = len(edgelines) - 1 if pd.isna(labels.max()) else int(labels.max())
    filling = pd.Series(range(max_label + 1, max_label + len(edgelines) + 1))
    labels = labels.fillna(filling)

    return labels.values


def weld_edges(
    edgelines: list | np.ndarray | gpd.GeoSeries,
    *,
    ignore: None | gpd.GeoSeries = None,
) -> list | np.ndarray | gpd.GeoSeries:
    """Combine lines sharing an endpoint (if only 2 lines share that point).
    Lightweight version of ``remove_interstitial_nodes()``.

    Parameters
    ----------
    edgelines : list | np.ndarray | gpd.GeoSeries
        Collection of line objects.
    ignore : None | gpd.GeoSeries = None
        Nodes to ignore when welding components.

    Returns
    -------
    list | np.ndarray | gpd.GeoSeries
        Resultant welded ``edgelines`` if more than 1 passed in, otherwise
        the original ``edgelines`` object.
    """
    if len(edgelines) < 2:
        return edgelines
    labels = get_components(edgelines, ignore=ignore)
    return (
        gpd.GeoSeries(edgelines)
        .groupby(labels)
        .agg(lambda x: shapely.line_merge(shapely.GeometryCollection(x.values)))
    ).tolist()


def induce_nodes(streets: gpd.GeoDataFrame, *, eps: float = 1e-4) -> gpd.GeoDataFrame:
    """Adding potentially missing nodes on intersections of individual LineString
    endpoints with the remaining network. The idea behind is that if a line ends
    on an intersection with another, there should be a node on both of them.

    Parameters
    ----------
    streets : geopandas.GeoDataFrame
        Input LineString geometries.
    eps : float = 1e-4
        Tolerance epsilon for point snapping passed into ``nodes.split()``.

    Returns
    -------
    geopandas.GeoDataFrame
        Updated ``streets`` with (potentially) added nodes.
    """

    sindex_kws = {"predicate": "dwithin", "distance": 1e-4}

    # identify degree mismatch cases
    nodes_degree_mismatch = _identify_degree_mismatch(streets, sindex_kws)

    # ensure loop topology cases:
    #   - loop nodes intersecting non-loops
    #   - loop nodes intersecting other loops
    nodes_off_loops, nodes_on_loops = _makes_loop_contact(streets, sindex_kws)

    # all nodes to induce
    nodes_to_induce = pd.concat(
        [nodes_degree_mismatch, nodes_off_loops, nodes_on_loops]
    )

    return split(nodes_to_induce.geometry, streets, streets.crs, eps=eps)


def _identify_degree_mismatch(
    edges: gpd.GeoDataFrame, sindex_kws: dict
) -> gpd.GeoSeries:
    """Helper to identify difference of observed vs. expected node degree."""
    nodes = _nodes_degrees_from_edges(edges.geometry)
    nodes = nodes.set_crs(edges.crs)
    nix, eix = edges.sindex.query(nodes.geometry, **sindex_kws)
    coo_vals = ([True] * len(nix), (nix, eix))
    coo_shape = (len(nodes), len(edges))
    intersects = sparse.coo_array(coo_vals, shape=coo_shape, dtype=np.bool_)
    nodes["expected_degree"] = intersects.sum(axis=1)
    return nodes[nodes["degree"] != nodes["expected_degree"]].geometry


def _nodes_from_edges(
    edgelines: list | np.ndarray | gpd.GeoSeries,
    return_degrees=False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Helper to get network nodes from edges' geometries."""
    edgelines = np.array(edgelines)
    start_points = shapely.get_point(edgelines, 0)
    end_points = shapely.get_point(edgelines, -1)
    node_coords = np.unique(
        shapely.get_coordinates(np.concatenate([start_points, end_points])),
        axis=0,
        return_counts=return_degrees,
    )
    if return_degrees:
        node_coords, degrees = node_coords
    node_points = shapely.points(node_coords)
    if return_degrees:
        return node_points, degrees
    else:
        return node_points


def _nodes_degrees_from_edges(
    edgelines: list | np.ndarray | gpd.GeoSeries,
) -> gpd.GeoDataFrame:
    """Helper to get network nodes and their degrees from edges' geometries."""
    node_points, degrees = _nodes_from_edges(edgelines, return_degrees=True)
    nodes_gdf = gpd.GeoDataFrame({"degree": degrees, "geometry": node_points})
    return nodes_gdf


def _makes_loop_contact(
    edges: gpd.GeoDataFrame, sindex_kws: dict
) -> tuple[gpd.GeoSeries, gpd.GeoSeries]:
    """Helper to identify:
    1. loop nodes intersecting non-loops
    2. loop nodes intersecting other loops
    """

    loops, not_loops = _loops_and_non_loops(edges)
    loop_points = shapely.points(loops.get_coordinates().values)
    loop_gdf = gpd.GeoDataFrame(geometry=loop_points, crs=edges.crs)
    loop_point_geoms = loop_gdf.geometry

    # loop points intersecting non-loops
    nodes_from_non_loops_ix, _ = not_loops.sindex.query(loop_point_geoms, **sindex_kws)

    # loop points intersecting other loops
    nodes_from_loops_ix, _ = loops.sindex.query(loop_point_geoms, **sindex_kws)
    loop_x_loop, n_loop_x_loop = np.unique(nodes_from_loops_ix, return_counts=True)
    nodes_from_loops_ix = loop_x_loop[n_loop_x_loop > 1]

    # tease out both varieties
    nodes_non_loops = loop_gdf.loc[nodes_from_non_loops_ix]
    nodes_loops = loop_gdf.loc[nodes_from_loops_ix]

    return nodes_non_loops.geometry, nodes_loops.geometry


def _loops_and_non_loops(
    edges: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Bifurcate edge gdf into loops and non-loops."""
    loop_mask = edges.is_ring
    not_loops = edges[~loop_mask]
    loops = edges[loop_mask]
    return loops, not_loops


def remove_interstitial_nodes(
    gdf: gpd.GeoSeries | gpd.GeoDataFrame, *, aggfunc: str | dict = "first", **kwargs
) -> gpd.GeoSeries | gpd.GeoDataFrame:
    """Clean topology of existing LineString geometry by removal of nodes of degree 2.

    Returns the original gdf if there’s no node of degree 2.

    Parameters
    ----------
    gdf : geopandas.GeoSeries | geopandas.GeoDataFrame
        Input edgelines process. If any edges are ``MultiLineString`` they
        will be exploded into constituent ``LineString`` components.
    aggfunc : str | dict = 'first'
        Aggregate function for processing non-spatial component.
    **kwargs
        Keyword arguments for ``aggfunc``.

    Returns
    -------
    geopandas.GeoSeries | geopandas.GeoDataFrame
       The original input ``gdf`` if only 1 edgeline, otherwise the processed
       edgeline without interstitial nodes.

    Notes
    -----
    Any 3D geometries are (potentially) downcast in loops.
    """

    def merge_geometries(block: gpd.GeoSeries) -> shapely.LineString:
        """Helper in processing the spatial component."""
        return shapely.line_merge(shapely.GeometryCollection(block.values))

    if len(gdf) < 2:
        return gdf

    if isinstance(gdf, gpd.GeoSeries):
        gdf = gdf.to_frame("geometry")

    gdf = gdf.explode(ignore_index=True)

    labels = get_components(gdf.geometry)

    # Process non-spatial component
    data = gdf.drop(labels=gdf.geometry.name, axis=1)
    aggregated_data = data.groupby(by=labels).agg(aggfunc, **kwargs)
    aggregated_data.columns = aggregated_data.columns.to_flat_index()

    # Process spatial component
    g = gdf.groupby(group_keys=False, by=labels)[gdf.geometry.name].agg(
        merge_geometries
    )
    aggregated_geometry = gpd.GeoDataFrame(g, geometry=gdf.geometry.name, crs=gdf.crs)

    # Recombine
    aggregated = aggregated_geometry.join(aggregated_data)

    # Derive nodes
    nodes = _nodes_from_edges(aggregated.geometry)
    # Bifurcate edges into loops and non-loops
    loops, not_loops = _loops_and_non_loops(aggregated)

    # Ensure:
    #   - all loops have exactly 1 endpoint; and
    #   - that endpoint shares a node with an intersecting line
    fixed_loops = []
    fixed_index = []
    node_ix, loop_ix = loops.sindex.query(nodes, predicate="intersects")
    for ix in np.unique(loop_ix):
        loop_geom = loops.geometry.iloc[ix]
        target_nodes = nodes[node_ix[loop_ix == ix]]
        if len(target_nodes) == 2:
            new_sequence = _rotate_loop_coords(loop_geom, not_loops)
            fixed_loops.append(new_sequence)
            fixed_index.append(ix)

    aggregated.loc[loops.index[fixed_index], aggregated.geometry.name] = fixed_loops
    return aggregated.reset_index(drop=True)


def _rotate_loop_coords(
    loop_geom: shapely.LineString, not_loops: gpd.GeoDataFrame
) -> np.ndarray:
    """Rotate loop node coordinates if needed to ensure topology.

    The function is prone to errors with super weird configurations.
    If it fails, return the original to avoid breaking the entire workflow.
    """
    try:
        loop_coords = shapely.get_coordinates(loop_geom)
        loop_points = gpd.GeoDataFrame(geometry=shapely.points(loop_coords))
        loop_points_ix, _ = not_loops.sindex.query(
            loop_points.geometry, predicate="dwithin", distance=1e-4
        )

        mode = loop_points.loc[loop_points_ix].geometry.mode()

        # if there is a non-planar intersection, we may have multiple points. Check with
        # entrypoints only in that case
        if mode.shape[0] > 1:
            loop_points_ix, _ = not_loops.sindex.query(
                loop_points.geometry, predicate="dwithin", distance=1e-4
            )
            new_mode = loop_points.loc[loop_points_ix].geometry.mode()
            # if that did not help, just pick one to avoid failure and hope for the best
            if new_mode.empty | new_mode.shape[0] > 1:
                mode = mode.iloc[[0]]

        new_start = mode.get_coordinates().values
        _coords_match = (loop_coords == new_start).all(axis=1)
        new_start_idx = np.where(_coords_match)[0].squeeze()

        rolled_coords = np.roll(loop_coords[:-1], -new_start_idx, axis=0)
        new_sequence = np.append(rolled_coords, rolled_coords[[0]], axis=0)
        return shapely.LineString(new_sequence)

    except ValueError:
        warnings.warn(
            f"Loop at {loop_geom.centroid} could not be re-ordered. "
            "Topology might be suboptimal.",
            stacklevel=3,
        )
        return loop_geom


def fix_topology(
    streets: gpd.GeoDataFrame,
    *,
    eps: float = 1e-4,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Fix street network topology. This ensures correct topology of the network by:

        1.  Adding potentially missing nodes...
                on intersections of individual LineString endpoints
                with the remaining network. The idea behind is that
                if a line ends on an intersection with another, there
                should be a node on both of them.
        2. Removing nodes of degree 2...
                that have no meaning in the network used within our framework.
        3. Removing duplicated geometries (irrespective of orientation).

    Parameters
    ----------
    streets : geopandas.GeoDataFrame
        Input LineString geometries.
    eps : float = 1e-4
        Tolerance epsilon for point snapping passed into ``nodes.split()``.
    **kwargs : dict
        Key word arguments passed into ``remove_interstitial_nodes()``.

    Returns
    -------
    gpd.GeoDataFrame
        The input streets that now have fixed topology and are ready
        to proceed through the simplification algorithm.
    """
    streets = streets[~streets.geometry.normalize().duplicated()].copy()
    streets_w_nodes = induce_nodes(streets, eps=eps)
    return remove_interstitial_nodes(streets_w_nodes, **kwargs)


def consolidate_nodes(
    gdf: np.ndarray | gpd.GeoSeries | gpd.GeoDataFrame,
    *,
    tolerance: float = 2.0,
    preserve_ends: bool = False,
) -> gpd.GeoSeries:
    """Return geometry with consolidated nodes.

    Replace clusters of nodes with a single node (weighted centroid
    of a cluster) and snap linestring geometry to it. Cluster is
    defined using hierarchical clustering with average linkage
    on coordinates cut at a cophenetic distance equal to ``tolerance``.

    The use of hierachical clustering avoids the chaining effect of a sequence
    of intersections within ``tolerance`` from each other that would happen with
    DBSCAN and similar solutions.

    Parameters
    ----------
    gdf : numpy.ndarray | geopandas.GeoSeries | geopandas.GeoDataFrame
        LineString geometries (usually representing street network).
    tolerance : float = 2.0
        The maximum distance between two nodes for one to be considered
        as in the neighborhood of the other. Nodes within tolerance are
        considered a part of a single cluster and will be consolidated.
    preserve_ends : bool = False
        If ``True``, nodes of a degree 1 will be excluded from the consolidation.

    Returns
    -------
    geopandas.GeoSeries
        Updated input ``gdf`` of LineStrings with consolidated nodes.
    """

    def _get_labels(nodes: gpd.GeoDataFrame) -> np.ndarray:
        """Generate node cluster labels that avoids a chaining effect."""
        linkage = hierarchy.linkage(shapely.get_coordinates(nodes), method="average")
        labels = (
            hierarchy.fcluster(linkage, tolerance, criterion="distance").astype(str)
            + f"_{nodes.name}"
        )
        return labels

    if not isinstance(gdf, gpd.GeoDataFrame):
        gdf = gpd.GeoDataFrame(geometry=gdf)

    nodes = _nodes_degrees_from_edges(gdf.geometry)

    if preserve_ends:
        # keep at least one meter of original geometry around each end
        ends = nodes[nodes["degree"] == 1].buffer(1)
        nodes = nodes[nodes["degree"] > 1].copy()

        # if all we have are ends, return the original
        # - this is generally when called from within ``geometry._consolidate()``
        if nodes.shape[0] < 2:
            gdf["_status"] = "original"
            return gdf

    # Get node cluster to be consolidated. First, get components of possible clusters
    # then do the linkage itself. Otherwise, it's dead slow and needs a ton of memory.
    db = DBSCAN(eps=tolerance, min_samples=2).fit(nodes.get_coordinates())
    comp_labels = db.labels_
    mask = comp_labels > -1
    components = comp_labels[mask]
    nodes_to_merge = nodes[mask]

    # get grouped node cluster labels to determines which nodes must change
    grouped = (
        pd.Series(nodes_to_merge.geometry).groupby(components).transform(_get_labels)
    )
    nodes["lab"] = grouped
    unique, counts = np.unique(nodes["lab"].dropna(), return_counts=True)
    actual_clusters = unique[counts > 1]
    change = nodes[nodes["lab"].isin(actual_clusters)]

    # no change needed, return the original
    if change.empty:
        gdf["_status"] = "original"
        return gdf

    gdf = gdf.copy()
    geom = gdf.geometry.copy()
    status = pd.Series("original", index=geom.index)

    # loop over clusters, cut out geometry within tolerance / 2 and replace it
    # with spider-like geometry to the weighted centroid of a cluster
    spiders = []
    midpoints = []

    clusters = change.dissolve(change["lab"])

    # TODO: not optimal but avoids some MultiLineStrings but not all
    cookies = clusters.buffer(tolerance / 2).convex_hull

    if preserve_ends:
        cookies = cookies.to_frame().overlay(ends.to_frame(), how="difference")

    for cluster, cookie in zip(clusters.geometry, cookies.geometry, strict=True):
        inds = geom.sindex.query(cookie, predicate="intersects")
        pts = shapely.get_coordinates(geom.iloc[inds].intersection(cookie.boundary))
        if pts.shape[0] > 0:
            # TODO: this may result in MultiLineString - we need to avoid that
            # TODO: It is temporarily fixed by that explode in return
            geom.iloc[inds] = geom.iloc[inds].difference(cookie)

            status.iloc[inds] = "changed"
            midpoint = np.mean(shapely.get_coordinates(cluster), axis=0)
            midpoints.append(midpoint)
            mids = np.array([midpoint] * len(pts))

            spider = shapely.linestrings(
                np.array([pts[:, 0], mids[:, 0]]).T,
                y=np.array([pts[:, 1], mids[:, 1]]).T,
            )
            spiders.append(spider)

    gdf = gdf.set_geometry(geom)
    gdf["_status"] = status

    if spiders:
        # combine geometries
        geoms = np.hstack(spiders)
        gdf = pd.concat([gdf, gpd.GeoDataFrame(geometry=geoms, crs=geom.crs)])

    agg: dict[str, str | typing.Callable] = {"_status": _status}
    for c in gdf.columns.drop(gdf.active_geometry_name):
        if c != "_status":
            agg[c] = "first"
    return remove_interstitial_nodes(
        gdf[~gdf.geometry.is_empty].explode(),
        # NOTE: this aggfunc needs to be able to process all the columns
        aggfunc=agg,
    )
