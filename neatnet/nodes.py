import typing

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import pyproj
import shapely
from scipy import sparse


def split(
    split_points: list | np.ndarray | gpd.GeoSeries,
    cleaned_roads: gpd.GeoDataFrame,
    crs: str | pyproj.CRS,
    eps: float = 1e-4,
) -> gpd.GeoSeries | gpd.GeoDataFrame:
    """Split lines on new nodes.

    Parameters
    ----------
    split_points : list | numpy.ndarray
        Points to split the ``cleaned_roads``.
    cleaned_roads : geopandas.GeoDataFrame
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
        _, ix = cleaned_roads.sindex.nearest(split, max_distance=eps)
        row = cleaned_roads.iloc[ix]
        edge = row.geometry
        if edge.shape[0] == 1:
            row = row.iloc[0]
            lines_split = _snap_n_split(edge.item(), split, eps)
            if lines_split.shape[0] > 1:
                gdf_split = gpd.GeoDataFrame(geometry=lines_split, crs=crs)
                for c in row.index.drop(["geometry", "_status"], errors="ignore"):
                    gdf_split[c] = row[c]
                gdf_split["_status"] = "changed"
                cleaned_roads = pd.concat(
                    [cleaned_roads.drop(edge.index[0]), gdf_split],
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
                cleaned_roads = pd.concat(
                    [cleaned_roads.drop(to_be_dropped), gdf_split],
                    ignore_index=True,
                )
                cleaned_roads = gpd.GeoDataFrame(
                    cleaned_roads, geometry="geometry", crs=crs
                )

    return cleaned_roads.reset_index(drop=True)


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


def get_components(
    edgelines: list | np.ndarray | gpd.GeoSeries,
    ignore: None | gpd.GeoSeries = None,
) -> np.ndarray:
    """Associate edges with connected component labels and return.

    Parameters
    ----------
    edgelines : list | np.ndarray | gpd.GeoSeries
        Collection of line objects.
    ignore : None | gpd.GeoSeries = None
        Nodes to ignore when labeling components.

    Returns
    -------
    np.ndarray
        Edge connected component labels.

    Notes
    -----
    See [https://github.com/uscuni/neatnet/issues/56] for detailed explanation of
    output.
    """
    edgelines = np.array(edgelines)
    start_points = shapely.get_point(edgelines, 0)
    end_points = shapely.get_point(edgelines, -1)
    points = shapely.points(
        np.unique(
            shapely.get_coordinates(np.concatenate([start_points, end_points])), axis=0
        )
    )
    if ignore is not None:
        mask = np.isin(points, ignore)
        points = points[~mask]
    # query LineString geometry to identify points intersecting 2 geometries
    inp, res = shapely.STRtree(shapely.boundary(edgelines)).query(
        points, predicate="intersects"
    )
    unique, counts = np.unique(inp, return_counts=True)
    mask = np.isin(inp, unique[counts == 2])
    merge_res = res[mask]
    merge_inp = inp[mask]
    closed = np.arange(len(edgelines))[shapely.is_closed(edgelines)]
    mask = np.isin(merge_res, closed) | np.isin(merge_inp, closed)
    merge_res = merge_res[~mask]
    merge_inp = merge_inp[~mask]
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
    ignore: None | gpd.GeoSeries = None,
) -> list | np.ndarray | gpd.GeoSeries:
    """Combine lines sharing an endpoint (if only 2 lines share that point).
    Lightweight version of ``remove_false_nodes()``.

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


def induce_nodes(roads: gpd.GeoDataFrame, eps: float = 1e-4) -> gpd.GeoDataFrame:
    """Adding potentially missing nodes on intersections of individual LineString
    endpoints with the remaining network. The idea behind is that if a line ends
    on an intersection with another, there should be a node on both of them.

    Parameters
    ----------
    roads : geopandas.GeoDataFrame
        Input LineString geometries.
    eps : float = 1e-4
        Tolerance epsilon for point snapping passed into ``nodes.split()``.

    Returns
    -------
    geopandas.GeoDataFrame
        Updated ``roads`` with (potentially) added nodes.
    """

    sindex_kws = {"predicate": "dwithin", "distance": 1e-4}

    # identify degree mismatch cases
    nodes_degree_mismatch = _identify_degree_mismatch(roads, sindex_kws)

    # ensure loop topology cases:
    #   - loop nodes intersecting non-loops
    #   - loop nodes intersecting other loops
    nodes_off_loops, nodes_on_loops = _makes_loop_contact(roads, sindex_kws)

    # all nodes to induce
    nodes_to_induce = pd.concat(
        [nodes_degree_mismatch, nodes_off_loops, nodes_on_loops]
    )

    return split(nodes_to_induce.geometry, roads, roads.crs, eps=eps)


def _identify_degree_mismatch(
    edges: gpd.GeoDataFrame, sindex_kws: dict
) -> gpd.GeoSeries:
    """Helper to identify difference of observed vs. expected node degree."""
    nodes = momepy.nx_to_gdf(momepy.node_degree(momepy.gdf_to_nx(edges)), lines=False)
    nix, eix = edges.sindex.query(nodes.geometry, **sindex_kws)
    coo_vals = ([True] * len(nix), (nix, eix))
    coo_shape = (len(nodes), len(edges))
    intersects = sparse.coo_array(coo_vals, shape=coo_shape, dtype=np.bool_)
    nodes["expected_degree"] = intersects.sum(axis=1)
    return nodes[nodes["degree"] != nodes["expected_degree"]].geometry


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


def remove_false_nodes(
    gdf: gpd.GeoSeries | gpd.GeoDataFrame, aggfunc: str | dict = "first", **kwargs
) -> gpd.GeoSeries | gpd.GeoDataFrame:
    """Reimplementation of ``momepy.remove_false_nodes()`` that preserves attributes.

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
    nodes = momepy.nx_to_gdf(
        momepy.node_degree(momepy.gdf_to_nx(aggregated[[aggregated.geometry.name]])),
        lines=False,
    )

    # Bifurcate edges into loops and non-loops
    loops, not_loops = _loops_and_non_loops(aggregated)

    # Ensure:
    #   - all loops have exactly 1 endpoint; and
    #   - that endpoint shares a node with an intersecting line
    fixed_loops = []
    fixed_index = []
    node_ix, loop_ix = loops.sindex.query(nodes.geometry, predicate="intersects")
    for ix in np.unique(loop_ix):
        loop_geom = loops.geometry.iloc[ix]
        target_nodes = nodes.geometry.iloc[node_ix[loop_ix == ix]]
        if len(target_nodes) == 2:
            new_sequence = _rotate_loop_coords(loop_geom, not_loops)
            fixed_loops.append(shapely.LineString(new_sequence))
            fixed_index.append(ix)

    aggregated.loc[loops.index[fixed_index], aggregated.geometry.name] = fixed_loops
    return aggregated.reset_index(drop=True)


def _rotate_loop_coords(
    loop_geom: shapely.LineString, not_loops: gpd.GeoDataFrame
) -> np.ndarray:
    """Rotate loop node coordinates if needed to ensure topology."""

    loop_coords = shapely.get_coordinates(loop_geom)
    loop_points = gpd.GeoDataFrame(geometry=shapely.points(loop_coords))
    loop_points_ix, _ = not_loops.sindex.query(
        loop_points.geometry, predicate="dwithin", distance=1e-4
    )

    new_start = loop_points.loc[loop_points_ix].geometry.mode().get_coordinates().values
    _coords_match = (loop_coords == new_start).all(axis=1)
    new_start_idx = np.where(_coords_match)[0].squeeze()

    rolled_coords = np.roll(loop_coords[:-1], -new_start_idx, axis=0)
    new_sequence = np.append(rolled_coords, rolled_coords[[0]], axis=0)
    return new_sequence


def fix_topology(
    roads: gpd.GeoDataFrame,
    eps: float = 1e-4,
    **kwargs,
) -> gpd.GeoDataFrame:
    """Fix road network topology. This ensures correct topology of the network by:

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
    roads : geopandas.GeoDataFrame
        Input LineString geometries.
    eps : float = 1e-4
        Tolerance epsilon for point snapping passed into ``nodes.split()``.
    **kwargs : dict
        Key word arguments passed into ``remove_false_nodes()``.

    Returns
    -------
    gpd.GeoDataFrame
        The input roads that now have fixed topology and are ready
        to proceed through the simplification algorithm.
    """
    roads = roads[~roads.geometry.normalize().duplicated()].copy()
    roads_w_nodes = induce_nodes(roads, eps=eps)
    return remove_false_nodes(roads_w_nodes, **kwargs)


def consolidate_nodes(
    gdf: gpd.GeoDataFrame,
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
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with LineStrings (usually representing street network).
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
    from scipy.cluster import hierarchy

    if isinstance(gdf, gpd.GeoSeries):
        gdf = gdf.to_frame("geometry")
    elif isinstance(gdf, np.ndarray):
        gdf = gpd.GeoDataFrame(geometry=gdf)

    nodes = momepy.nx_to_gdf(momepy.node_degree(momepy.gdf_to_nx(gdf)), lines=False)

    if preserve_ends:
        # keep at least one meter of original geometry around each end
        ends = nodes[nodes["degree"] == 1].buffer(1)
        nodes = nodes[nodes["degree"] > 1].copy()

        # if all we have are ends, return the original
        # - this is generally when called from within ``geometry._consolidate()``
        if nodes.shape[0] < 2:
            gdf["_status"] = "original"
            return gdf

    # get clusters of nodes which should be consolidated
    linkage = hierarchy.linkage(nodes.get_coordinates(), method="average")
    nodes["lab"] = hierarchy.fcluster(linkage, tolerance, criterion="distance")
    unique, counts = np.unique(nodes["lab"], return_counts=True)
    actual_clusters = unique[counts > 1]
    change = nodes[nodes["lab"].isin(actual_clusters)]

    # no change needed, return the original
    if change.empty:
        gdf["_status"] = "original"
        return gdf

    gdf = gdf.copy()
    # get geometry
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
    return remove_false_nodes(
        gdf[~gdf.geometry.is_empty].explode(),
        # NOTE: this aggfunc needs to be able to process all the columns
        aggfunc=agg,
    )
