import logging
import typing
import warnings

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import shapely
from esda import shape
from libpysal import graph
from scipy import sparse

from .geometry import (
    _is_within,
    angle_between_two_lines,
    snap_to_targets,
    voronoi_skeleton,
)
from .nodes import weld_edges

logger = logging.getLogger(__name__)


def get_artifacts(
    roads: gpd.GeoDataFrame,
    threshold: None | float | int = None,
    threshold_fallback: None | float | int = None,
    area_threshold_blocks: float | int = 1e5,
    isoareal_threshold_blocks: float | int = 0.5,
    area_threshold_circles: float | int = 5e4,
    isoareal_threshold_circles_enclosed: float | int = 0.75,
    isoperimetric_threshold_circles_touching: float | int = 0.9,
    exclusion_mask: None | gpd.GeoSeries = None,
    predicate: str = "intersects",
) -> tuple[gpd.GeoDataFrame, float]:
    """Extract face artifacts and return the FAI threshold.
    See :cite:`fleischmann2023` for more details.

    Parameters
    ----------
    roads : geopandas.GeoDataFrame
        Input roads that have been preprocessed.
    threshold : None | float | int = None
        First option threshold used to determine face artifacts. See the
        ``artifact_threshold`` keyword argument in ``simplify.simplify_network()``.
    threshold_fallback : None | float | int = None
        Second option threshold used to determine face artifacts. See the
        ``artifact_threshold_fallback`` keyword argument in
        ``simplify.simplify_network()``.
    area_threshold_blocks : float | int = 1e5
        This is the first threshold for detecting block-like artifacts whose
        Face Artifact Index (see :cite:`fleischmann2023`) is above the value
        passed in ``artifact_threshold``.
        If a polygon has an area below ``area_threshold_blocks``, *and*
        is of elongated shape (see also ``isoareal_threshold_blocks``),
        *and* touches at least one polygon that has already been classified as artifact,
        then it will be classified as an artifact.
    isoareal_threshold_blocks : float | int = 0.5
        This is the second threshold for detecting block-like artifacts whose
        Face Artifact Index (see :cite:`fleischmann2023`) is above the value
        passed in ``artifact_threshold``. If a polygon has an isoareal quotient
        below ``isoareal_threshold_blocks`` (see ``esda.shape.isoareal_quotient``),
        i.e., if it has an elongated shape; *and* it has a sufficiently small area
        (see also ``area_threshold_blocks``), *and* if it touches at least one
         polygon that has already been detected as an artifact,
        then it will be classified as an artifact.
    area_threshold_circles : float | int = 5e4
        This is the first threshold for detecting circle-like artifacts whose
        Face Artifact Index (see :cite:`fleischmann2023`) is above the value
        passed in ``artifact_threshold``. If a polygon has an area below
        ``area_threshold_circles``, *and* one of the following 2 cases is given:
        (a) the polygon is touched, but not enclosed by polygons already classified
        as artifacts, *and* with an isoperimetric quotient
        (see ``esda.shape.isoperimetric_quotient``)
        above ``isoperimetric_threshold_circles_touching``, i.e., if its shape
        is close to circular; or (b) the polygon is fully enclosed by polygons
        already classified as artifacts, *and* with an isoareal quotient
        above
        ``isoareal_threshold_circles_enclosed``, i.e., if its shape is
        close to circular; then it will be classified as an artifact.
    isoareal_threshold_circles_enclosed : float | int = 0.75
        This is the second threshold for detecting circle-like artifacts whose
        Face Artifact Index (see :cite:`fleischmann2023`) is above the value
        passed in ``artifact_threshold``. If a polygon has a sufficiently small
        area (see also ``area_threshold_circles``), *and* the polygon is
        fully enclosed by polygons already classified as artifacts,
        *and* its isoareal quotient (see ``esda.shape.isoareal_quotient``)
        is above the value passed to ``isoareal_threshold_circles_enclosed``,
        i.e., if its shape is close to circular;
        then it will be classified as an artifact.
    isoperimetric_threshold_circles_touching : float | int = 0.9
        This is the third threshold for detecting circle-like artifacts whose
        Face Artifact Index (see :cite:`fleischmann2023`)
        is above the value passed in ``artifact_threshold``.
        If a polygon has a sufficiently small area
        (see also ``area_threshold_circles``), *and* the polygon is touched
        by at least one polygon already classified as artifact,
        *and* its isoperimetric quotient (see ``esda.shape.isoperimetric_quotient``)
        is above the value passed to ``isoperimetric_threshold_circles_touching``,
        i.e., if its shape is close to circular;
        then it will be classified as an artifact.
    exclusion_mask : None | gpd.GeoSeries = None
        Polygons used to determine face artifacts to exclude from returned output.
    predicate : str = 'intersects'
        The spatial predicate used to exclude face artifacts from returned output.

    Returns
    -------
    artifacts : geopandas.GeoDataFrame
        Face artifact polygons.
    threshold : float
        Resultant artifact detection threshold from ``momepy.FaceArtifacts.threshold``.
        May also be the returned value of ``threshold`` or ``threshold_fallback``.
    """

    def _relate(neighs: tuple, cond: typing.Callable) -> bool:
        """Helper for relating artifacts."""
        return len(neighs) > 0 and cond(polys.loc[list(neighs), "is_artifact"])

    with warnings.catch_warnings():  # the second loop likey won't find threshold
        warnings.filterwarnings("ignore", message="No threshold found")
        fas = momepy.FaceArtifacts(roads)
    polys = fas.polygons.set_crs(roads.crs)

    # rook neighbors
    rook = graph.Graph.build_contiguity(polys, rook=True)
    polys["neighbors"] = rook.neighbors

    # polygons are not artifacts...
    polys["is_artifact"] = False
    # ...unless the fai is below the threshold
    if threshold is None:
        if not fas.threshold and threshold_fallback:
            threshold = threshold_fallback
        elif not fas.threshold and not threshold_fallback:
            raise ValueError(
                "No threshold for artifact detection found. Pass explicit "
                "`threshold` or `threshold_fallback` to provide the value directly."
            )
        else:
            threshold = fas.threshold
    polys.loc[polys["face_artifact_index"] < threshold, "is_artifact"] = True

    # compute area, isoareal quotient, and isoperimetric quotient
    polys["area_sqm"] = polys.area
    polys["isoareal_index"] = shape.isoareal_quotient(polys.geometry)
    polys["isoperimetric_quotient"] = shape.isoperimetric_quotient(polys.geometry)

    is_artifact = polys["is_artifact"]
    area = polys["area_sqm"]
    isoareal = polys["isoareal_index"]
    isoperimetric = polys["isoperimetric_quotient"]

    # iterate to account for artifacts that become
    # enclosed or touching by new designation
    while True:
        # count number of artifacts to break while loop
        # when no new artifacts are added
        artifact_count_before = sum(is_artifact)

        # polygons that are enclosed by artifacts (at this moment)
        polys["enclosed"] = polys.apply(lambda x: _relate(x["neighbors"], all), axis=1)

        # polygons that are touching artifacts (at this moment)
        polys["touching"] = polys.apply(lambda x: _relate(x["neighbors"], any), axis=1)

        # "block" like artifacts (that are not too big or too rectangular)
        # TODO: there are still some dual carriageway - type blocks
        # TODO: that slip through this one
        cond_geom = polys["enclosed"] | polys["touching"]
        cond_area = area < area_threshold_blocks
        cond_metric = isoareal < isoareal_threshold_blocks
        polys.loc[cond_geom & cond_area & cond_metric, "is_artifact"] = True

        # "circle" like artifacts (that are small and quite circular)
        # -- circles enclosed
        cond_geom = polys["enclosed"]
        cond_area = area < area_threshold_circles
        cond_metric = isoareal > isoareal_threshold_circles_enclosed
        polys.loc[cond_geom & cond_area & cond_metric, "is_artifact"] = True

        # -- circles touching
        cond_geom = polys["touching"]
        cond_area = area < area_threshold_circles
        cond_metric = isoperimetric > isoperimetric_threshold_circles_touching
        polys.loc[cond_geom & cond_area & cond_metric, "is_artifact"] = True

        artifact_count_after = sum(is_artifact)
        if artifact_count_after == artifact_count_before:
            break

    artifacts = polys[is_artifact][["geometry"]].reset_index(drop=True)
    artifacts["id"] = artifacts.index

    if exclusion_mask is not None:
        _, art_idx = artifacts.sindex.query(exclusion_mask, predicate=predicate)
        artifacts = artifacts.drop(np.unique(art_idx)).copy()

    return artifacts, threshold


def ccss_special_case(
    primes: gpd.GeoSeries,
    conts_groups: gpd.GeoDataFrame,
    highest_hierarchy: gpd.GeoDataFrame,
    relevant_nodes: gpd.GeoDataFrame,
    split_points: list,
) -> np.ndarray:
    """If there are primes on both ``C``s, connect them. If there's
    prime on one ``C``, connect it to the other ``C``. If there are
    no primes, get midpoints on ``C``s and connect those.

    Parameters
    ----------
    primes : geopandas.GeoSeries
        Nodes that are external to the artifacts, but need to be kept
    conts_groups : geopandas.GeoDataFrame
        All ``C`` labeled edges dissolved by connected component label.
    highest_hierarchy : geopandas.GeoDataFrame
        ``edges`` in the ``C`` continuity group – ``edges[~es_mask]``.
    relevant_targets : geopandas.GeoDataFrame
        The nodes forming the artifact.
    split_points : list
        Points to be used for topological corrections.

    Returns
    -------
    numpy.ndarray
        New linestrings for reconnections.
    """

    if primes.empty:
        # midpoints solution
        c0 = conts_groups.geometry.iloc[0]
        c1 = conts_groups.geometry.iloc[1]
        p0 = shapely.line_interpolate_point(c0, 0.5, normalized=True)
        p1 = shapely.line_interpolate_point(c1, 0.5, normalized=True)
        new_connections = [shapely.LineString([p0, p1])]
        split_points.append(p0)
        split_points.append(p1)

    # one prime, get shortest line to the other C
    elif primes.shape[0] == 1:
        no_prime_c = conts_groups[conts_groups.disjoint(primes.geometry.item())]
        sl = shapely.shortest_line(primes.geometry.item(), no_prime_c.geometry.item())
        new_connections = [sl]
        split_points.append(shapely.get_point(sl, -1))

    # multiple primes, connect two nearest on distinct Cs
    else:
        primes_on_c0 = primes[primes.intersects(conts_groups.geometry.iloc[0])]
        primes_on_c1 = primes[primes.intersects(conts_groups.geometry.iloc[1])]

        if primes_on_c0.empty:
            sl = shapely.shortest_line(
                conts_groups.geometry.iloc[0], primes_on_c1.union_all()
            )
            new_connections = [sl]
            split_points.append(shapely.get_point(sl, 0))
        elif primes_on_c1.empty:
            sl = shapely.shortest_line(
                primes_on_c0.union_all(), conts_groups.geometry.iloc[1]
            )
            new_connections = [sl]
            split_points.append(shapely.get_point(sl, -1))
        else:
            new_connections = [
                shapely.shortest_line(
                    primes_on_c0.union_all(), primes_on_c1.union_all()
                )
            ]

    # some nodes may have ended unconnected. Find them and reconnect them.
    combined_linework = pd.concat(
        [
            highest_hierarchy,
            gpd.GeoSeries(new_connections, crs=highest_hierarchy.crs),
        ]
    ).union_all()
    missing = relevant_nodes[relevant_nodes.disjoint(combined_linework)]
    new_connections.extend(
        shapely.shortest_line(missing.geometry, combined_linework).tolist()
    )

    return np.array(new_connections)


def filter_connections(
    primes: gpd.GeoSeries,
    relevant_targets: gpd.GeoDataFrame,
    conts_groups: gpd.GeoDataFrame,
    new_connections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The skeleton returns connections to all the nodes. We need to keep only
    some, if there are multiple connections to a single C. We don't touch the other.

    Parameters
    ----------
    primes : geopandas.GeoSeries
        Nodes that are external to the artifacts, but need to be kept
    relevant_targets : geopandas.GeoDataFrame
        The nodes forming the artifact.
    conts_groups : geopandas.GeoDataFrame
        All ``C`` labeled edges dissolved by connected component label.
    new_connections : numpy.ndarray
        New linestrings for reconnections.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - Updated ``new_connections``
        - Connections intersecting ``C``
        - Connections intersecting ``primes``
    """

    unwanted = []
    keeping = []
    conn_c = []
    conn_p = []
    if not primes.empty and not relevant_targets.empty:
        targets = pd.concat([primes.geometry, relevant_targets.geometry])
    elif primes.empty:
        targets = relevant_targets.geometry
    else:
        targets = primes.geometry
    for c in conts_groups.geometry:
        int_mask = shapely.intersects(new_connections, c)
        connections_intersecting_c = new_connections[int_mask]
        conn_c.append(connections_intersecting_c)
        if len(connections_intersecting_c) > 1:
            prime_mask = shapely.intersects(
                connections_intersecting_c, targets.union_all()
            )
            connections_intersecting_primes = connections_intersecting_c[prime_mask]
            conn_p.append(connections_intersecting_primes)
            # if there are multiple connections to a single C, drop them and keep only
            # the shortest one leading to prime
            if (
                len(connections_intersecting_c) > 1
                and len(connections_intersecting_primes) > 0
            ):
                lens = shapely.length(connections_intersecting_primes)
                unwanted.append(connections_intersecting_c)
                keeping.append(connections_intersecting_primes[[np.argmin(lens)]])

            # fork on two nodes on C
            elif len(connections_intersecting_c) > 1:
                lens = shapely.length(connections_intersecting_c)
                unwanted.append(connections_intersecting_c)

    if len(unwanted) > 0:
        wanted_mask = ~np.isin(new_connections, np.concatenate(unwanted))
        if len(keeping) > 0:
            new_connections = np.concatenate(
                [new_connections[wanted_mask], np.concatenate(keeping)]
            )
        else:
            new_connections = new_connections[wanted_mask]
    return (
        new_connections,
        np.concatenate(conn_c) if len(conn_c) > 0 else np.array([]),
        np.concatenate(conn_p) if len(conn_p) > 0 else np.array([]),
    )


def avoid_forks(
    highest_hierarchy: gpd.GeoDataFrame,
    new_connections: np.ndarray,
    relevant_targets: gpd.GeoDataFrame,
    artifact: gpd.GeoDataFrame,
    split_points: list,
) -> np.ndarray:
    """Multiple ``C``s that are not intersecting. Avoid forks on the ends of a
    Voronoi skeleton. If one goes to a relevant node, keep it. If not, remove
    both and replace with a new shortest connection.

    Parameters
    ----------
    highest_hierarchy : geopandas.GeoDataFrame
        ``edges`` in the ``C`` continuity group – ``edges[~es_mask]``.
    new_connections : numpy.ndarray
        New linestrings for reconnections.
    relevant_targets : geopandas.GeoDataFrame
        The nodes forming the artifact.
    artifact : geopandas.GeoDataFrame
        The polygonal representation of the artifact.
    split_points : list
        Points to be used for topological corrections.

    Returns
    -------
    np.ndarray
        ``new_connections`` with either 1 prong of the fork if it connects to
        a relevant node or a new short connection if not.
    """

    int_mask = shapely.intersects(new_connections, highest_hierarchy.union_all())
    targets_mask = shapely.intersects(new_connections, relevant_targets.union_all())
    new_connections = new_connections[(int_mask * targets_mask) | np.invert(int_mask)]
    cont_diss = highest_hierarchy.dissolve(highest_hierarchy.coins_group).geometry
    addition, splitters = snap_to_targets(
        new_connections,
        artifact.geometry,
        cont_diss[cont_diss.disjoint(shapely.union_all(new_connections))],
    )
    split_points.extend(splitters)
    new_connections = np.concatenate([new_connections, addition])

    return new_connections


def reconnect(
    conts_groups: gpd.GeoDataFrame,
    new_connections: np.ndarray,
    artifact: gpd.GeoDataFrame,
    split_points: list,
    eps: float,
) -> np.ndarray:
    """Check for disconnected Cs and reconnect.

    Parameters
    ----------
    conts_groups : geopandas.GeoDataFrame
        All ``C`` labeled edges dissolved by connected component label.
    new_connections : numpy.ndarray
        New linestrings for reconnections.
    artifact : geopandas.GeoDataFrame
        The polygonal representation of the artifact.
    split_points : list
        Points to be used for topological corrections.
    eps : float
        Small tolerance epsilon.

    Returns
    -------
    np.ndarray
        ``new_connections`` with additional edges.
    """

    new_connections_comps = graph.Graph.build_contiguity(
        gpd.GeoSeries(new_connections), rook=False
    ).component_labels
    new_components = gpd.GeoDataFrame(geometry=new_connections).dissolve(
        new_connections_comps
    )
    additions = []
    for c in conts_groups.geometry:
        mask = new_components.intersects(c.buffer(eps))
        if not mask.all():
            adds, splitters = snap_to_targets(
                new_components[~mask].geometry, artifact.geometry, [c]
            )
            additions.extend(adds)
            split_points.extend(splitters)
    if len(additions) > 0:
        new_connections = np.concatenate([new_connections, additions])

    return new_connections


def remove_dangles(
    new_connections: np.ndarray,
    artifact: gpd.GeoDataFrame,
    eps: float = 1e-4,
) -> np.ndarray:
    """Dropping lines can introduce dangling edges. Remove those.

    Parameters
    ----------
    new_connections : np.ndarray
        New linestrings for reconnections.
    artifact : geopandas.GeoDataFrame
        The polygonal representation of the artifact.
    eps : float = 1e-4
        Small tolerance epsilon.

    Returns
    -------
    np.ndarray
        ``new_connections`` without dangling edges.
    """

    new_connections = shapely.line_merge(new_connections)
    pts0 = shapely.get_point(new_connections, 0)
    pts1 = shapely.get_point(new_connections, -1)
    pts = shapely.buffer(np.concatenate([pts0, pts1]), eps)
    all_idx, pts_idx = shapely.STRtree(pts).query(
        np.concatenate([pts, [artifact.geometry.boundary]]),
        predicate="intersects",
    )
    data = [True] * len(all_idx)
    sp = sparse.coo_array((data, (pts_idx, all_idx)), shape=(len(pts), len(pts) + 1))
    dangles = pts[sp.sum(axis=1) == 1]
    new_connections = new_connections[
        shapely.disjoint(new_connections, shapely.union_all(dangles))
    ]
    return new_connections


def one_remaining(
    relevant_targets: gpd.GeoDataFrame,
    remaining_nodes: gpd.GeoDataFrame,
    artifact: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    es_mask: pd.Series,
    max_segment_length: float | int,
    split_points: list,
    clip_limit: float | int,
    consolidation_tolerance: float | int,
) -> gpd.GeoDataFrame:
    """Resolve situations where there is 1 highest hierarchy and 1
    remaining node. This function is called within ``artifacts.nx_gx()``:
        * first SUBRANCH of BRANCH 2:
            * relevant node targets exist
                * only one remaining node

    Parameters
    ----------
    relevant_targets : geopandas.GeoDataFrame
        The nodes forming the artifact.
    remaining_nodes : geopandas.GeoDataFrame
        Nodes associated with the artifact that are not in group ``C``.
    artifact : geopandas.GeoDataFrame
        The polygonal representation of the artifact.
    edges : geopandas.GeoDataFrame
        Line geometries forming the artifact.
    es_mask : pandas.Series
        A mask for ``edges`` in the ``E`` and ``S`` continuity groups.
    max_segment_length : float | int
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
    split_points : list
        Points to be used for topological corrections.
    clip_limit : float | int
        Following generation of the Voronoi linework in ``geometry.voronoi_skeleton()``,
        we clip to fit inside the polygon. To ensure we get a space to make proper
        topological connections from the linework to the actual points on the edge of
        the polygon, we clip using a polygon with a negative buffer of ``clip_limit``
        or the radius of maximum inscribed circle, whichever is smaller.
    consolidation_tolerance : float | int
        Tolerance passed to node consolidation within the
        ``geometry.voronoi_skeleton()``.

    Returns
    -------
    geopandas.GeoDataFrame
        Newly resolved edges. The ``split_points`` parameter is also updated inplace.
    """

    # find the nearest relevant target
    remaining_nearest, target_nearest = relevant_targets.sindex.nearest(
        remaining_nodes.geometry, return_all=False
    )
    # create a new connection as the shortest straight line
    new_connections = shapely.shortest_line(
        remaining_nodes.geometry.iloc[remaining_nearest].values,
        relevant_targets.geometry.iloc[target_nearest].values,
    )
    # check if the new connection is within the artifact
    connections_within = _is_within(new_connections, artifact.geometry, 0.1)
    # if it is not within, discard it and use the skeleton instead
    if not connections_within.all():
        logger.debug("CONDITION _is_within False")

        new_connections, splitters = voronoi_skeleton(
            edges[es_mask].geometry,  # use edges that are being dropped
            poly=artifact.geometry,
            snap_to=relevant_targets.geometry.iloc[target_nearest],  # snap to nearest
            max_segment_length=max_segment_length,
            buffer=clip_limit,  # TODO: figure out if we need this
            clip_limit=clip_limit,
            consolidation_tolerance=consolidation_tolerance,
        )
        split_points.extend(splitters)

    return remove_dangles(new_connections, artifact)


def multiple_remaining(
    edges: gpd.GeoDataFrame,
    es_mask: pd.Series,
    artifact: pd.DataFrame,
    max_segment_length: float | int,
    highest_hierarchy: gpd.GeoDataFrame,
    split_points: list,
    snap_to: gpd.GeoSeries,
    clip_limit: float | int,
    consolidation_tolerance: float | int,
) -> gpd.GeoDataFrame:
    """Resolve situations where there is 1 highest hierarchy and multiple
    remaining nodes. This function is called within ``artifacts.nx_gx()``:
        * second SUBRANCH of BRANCH 2:
            * relevant node targets exist
                * more than one remaining node
        * second SUBRANCH of BRANCH 3:
            * no target nodes - snapping to C
                * more than one remaining node

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Line geometries forming the artifact.
    es_mask : pandas.Series
        A mask for ``edges`` in the ``E`` and ``S`` continuity groups.
    artifact : geopandas.GeoDataFrame
        The polygonal representation of the artifact.
    max_segment_length : float | int
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
    highest_hierarchy : geopandas.GeoDataFrame
        ``edges`` in the ``C`` continuity group – ``edges[~es_mask]``.
    split_points : list
        Points to be used for topological corrections.
    snap_to : geopandas.GeoSeries
        Snap to these relevant node targets.
    clip_limit : float | int
        Following generation of the Voronoi linework in ``geometry.voronoi_skeleton()``,
        we clip to fit inside the polygon. To ensure we get a space to make proper
        topological connections from the linework to the actual points on the edge of
        the polygon, we clip using a polygon with a negative buffer of ``clip_limit``
        or the radius of maximum inscribed circle, whichever is smaller.
    consolidation_tolerance : float | int
        Tolerance passed to node consolidation within the
        ``geometry.voronoi_skeleton()``.

    Returns
    -------
    geopandas.GeoDataFrame
        Newly resolved edges. The ``split_points`` parameter is also updated inplace.
    """

    # use skeleton to ensure all nodes are naturally connected
    new_connections, splitters = voronoi_skeleton(
        edges[es_mask].geometry,  # use edges that are being dropped
        poly=artifact.geometry,
        snap_to=snap_to,  # snap to relevant node targets
        max_segment_length=max_segment_length,
        secondary_snap_to=highest_hierarchy.geometry,
        clip_limit=clip_limit,
        consolidation_tolerance=consolidation_tolerance,
    )
    split_points.extend(splitters)

    return remove_dangles(new_connections, artifact)


def one_remaining_c(
    remaining_nodes: gpd.GeoDataFrame,
    highest_hierarchy: gpd.GeoDataFrame,
    artifact: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    es_mask: pd.Series,
    max_segment_length: float | int,
    split_points: list,
    clip_limit: float | int,
    consolidation_tolerance: float | int = 10,
) -> np.ndarray:
    """Resolve situations where there is 1 highest hierarchy and 1 remaining node.
    This function is called within ``artifacts.nx_gx()``:
        * first SUBRANCH of BRANCH 3:
            * no target nodes - snapping to C
                *  only one remaining node

    Parameters
    ----------
    remaining_nodes : geopandas.GeoDataFrame
        Nodes associated with the artifact that are not in group ``C``.
    highest_hierarchy : geopandas.GeoDataFrame
        ``edges`` in the ``C`` continuity group – ``edges[~es_mask]``.
    artifact : geopandas.GeoDataFrame
        The polygonal representation of the artifact.
    edges : geopandas.GeoDataFrame
        Line geometries forming the artifact.
    es_mask : pandas.Series
        A mask for ``edges`` in the ``E`` and ``S`` continuity groups.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
    split_points : list
        Points to be used for topological corrections.
    clip_limit : float | int
        Following generation of the Voronoi linework in ``geometry.voronoi_skeleton()``,
        we clip to fit inside the polygon. To ensure we get a space to make proper
        topological connections from the linework to the actual points on the edge of
        the polygon, we clip using a polygon with a negative buffer of ``clip_limit``
        or the radius of maximum inscribed circle, whichever is smaller.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation within the
        ``geometry.voronoi_skeleton()``.

    Returns
    -------
    numpy.ndarray
        Newly resolved edges. The ``split_points`` parameter is also updated inplace.
    """

    # create a new connection as the shortest straight line to any C
    new_connections = shapely.shortest_line(
        remaining_nodes.geometry.values,
        highest_hierarchy.union_all(),
    )
    splitters = shapely.get_point(new_connections, -1)
    # check if the new connection is within the artifact
    connections_within = _is_within(new_connections, artifact.geometry, 0.1)
    # if it is not within, discard it and use the skeleton instead
    if not connections_within.all():
        logger.debug("CONDITION _is_within False")

        new_connections, splitters = voronoi_skeleton(
            edges[es_mask].geometry,  # use edges that are being dropped
            poly=artifact.geometry,
            snap_to=highest_hierarchy.dissolve("coins_group").geometry,  # snap to Cs
            max_segment_length=max_segment_length,
            clip_limit=clip_limit,
            consolidation_tolerance=consolidation_tolerance,
        )
    split_points.extend(splitters)

    return new_connections


def loop(
    edges: gpd.GeoDataFrame,
    es_mask: pd.Series,
    highest_hierarchy: gpd.GeoDataFrame,
    artifact: gpd.GeoDataFrame,
    max_segment_length: float | int,
    clip_limit: float | int,
    split_points: list,
    min_dangle_length: float | int,
    eps: float = 1e-4,
) -> list:
    """Replace an artifact formed by a loop with a single line formed
    by a subset of the Voronoi skeleton.

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Line geometries forming the artifact.
    es_mask : pandas.Series
        A mask for ``edges`` in the ``E`` and ``S`` continuity groups.
    highest_hierarchy : geopandas.GeoDataFrame
        ``edges`` in the ``C`` continuity group – ``edges[~es_mask]``.
    artifact : geopandas.GeoDataFrame
        The polygonal representation of the artifact.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
    clip_limit : float | int
        Following generation of the Voronoi linework in ``geometry.voronoi_skeleton()``,
        we clip to fit inside the polygon. To ensure we get a space to make proper
        topological connections from the linework to the actual points on the edge of
        the polygon, we clip using a polygon with a negative buffer of ``clip_limit``
        or the radius of maximum inscribed circle, whichever is smaller.
    split_points : list
        Points to be used for topological corrections.
    min_dangle_length : float | int
         The threshold for determining if linestrings are dangling slivers to be
         removed or not.
    eps : float = 1e-4
        Small tolerance epsilon.

    Returns
    -------
    to_add : list
        Linestring geometries to be added.
    """

    # check if we need to add a deadend to represent the space
    to_add = []
    dropped = edges[es_mask].geometry.item()
    segments = line_segments(dropped)

    # figure out if there's a snapping node
    # Get nodes on Cs
    bd_points = highest_hierarchy.boundary.explode()
    # Identify nodes on primes
    primes = bd_points[bd_points.duplicated()]
    if primes.empty:
        logger.debug("SNAP TO highest_hierarchy")
        snap_to = highest_hierarchy.dissolve("coins_group").geometry
    else:
        logger.debug("SNAP TO primes")
        snap_to = primes

    _possible_dangle, splitters = voronoi_skeleton(
        segments,  # use edges that are being dropped
        poly=artifact.geometry,
        snap_to=snap_to,
        max_segment_length=max_segment_length,
        clip_limit=clip_limit,
        consolidation_tolerance=0,
    )
    split_points.extend(splitters)

    possible_dangle = gpd.GeoDataFrame(
        geometry=_possible_dangle[shapely.disjoint(_possible_dangle, dropped)]
    )
    if not possible_dangle.empty:
        comps = graph.Graph.build_contiguity(
            possible_dangle.difference(snap_to.union_all().buffer(eps)), rook=False
        )
        if comps.n_components > 1:
            # NOTE: it is unclear to me what exactly should happen here. I believe that
            # there will be cases when we may want to keep multiple dangles. Now keeping
            # only one.
            logger.debug("LOOP components many")
            comp_labels = comps.component_labels.values
            longest_component = possible_dangle.dissolve(
                comps.component_labels
            ).length.idxmax()
            possible_dangle = possible_dangle[comp_labels == longest_component]

        dangle_coins = momepy.COINS(
            possible_dangle,
            flow_mode=True,
        ).stroke_gdf()
        candidate = dangle_coins.loc[dangle_coins.length.idxmax()].geometry

        if candidate.intersects(snap_to.union_all().buffer(eps)) and (
            candidate.length > min_dangle_length
        ):
            logger.debug("LOOP intersects and length > min_dangle_length")
            if not primes.empty:
                points = [
                    shapely.get_point(candidate, 0),
                    shapely.get_point(candidate, -1),
                ]
                distances = shapely.distance(points, highest_hierarchy.union_all())
                if distances.max() > min_dangle_length:
                    logger.debug("LOOP prime check passed")
                    to_add.append(candidate)
            else:
                to_add.append(candidate)

    return to_add


def n1_g1_identical(
    edges: gpd.GeoDataFrame,
    *,
    to_drop: list,
    to_add: list,
    geom: shapely.Polygon,
    max_segment_length: float | int = 1,
    min_dangle_length: float | int = 10,
    clip_limit: float | int = 2,
) -> None:
    """Determine lines within artifacts to drop & add when dealing with typologies
    of 1 node and 1 continuity group – ``{C, E, S}``.

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Line geometries forming the artifact.
    to_drop : list
        List collecting geometries to be dropped.
    to_add : list
        List collecting geometries to be added.
    geom : shapely.Polygon
        The polygonal representation of the artifact.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
    min_dangle_length : float | int = 10
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    clip_limit : None | float | int = 2
        Following generation of the Voronoi linework in ``geometry.voronoi_skeleton()``,
        we clip to fit inside the polygon. To ensure we get a space to make proper
        topological connections from the linework to the actual points on the edge of
        the polygon, we clip using a polygon with a negative buffer of ``clip_limit``
        or the radius of maximum inscribed circle, whichever is smaller.

    Returns
    -------
    None
        ``to_drop`` and ``to_add`` are updated inplace.
    """

    to_drop.append(edges.index[0])
    dropped = edges.geometry.item()
    segments = line_segments(dropped)

    snap_to = shapely.get_point(dropped, 0)

    possible_dangle, _ = voronoi_skeleton(
        segments,  # use edges that are being dropped
        poly=geom,
        snap_to=[snap_to],
        max_segment_length=max_segment_length,
        clip_limit=clip_limit,
    )
    disjoint = shapely.disjoint(possible_dangle, dropped)
    connecting = shapely.intersects(possible_dangle, snap_to)
    dangle = possible_dangle[disjoint | connecting]

    dangle_geoms = gpd.GeoSeries(shapely.line_merge(dangle)).explode()
    dangle_coins = momepy.COINS(
        dangle_geoms, flow_mode=True, angle_threshold=120
    ).stroke_attribute()
    strokes = gpd.GeoDataFrame({"coin": dangle_coins}, geometry=dangle_geoms).dissolve(
        "coin"
    )
    entry = strokes.geometry[strokes.intersects(snap_to)].item()
    if entry.length > min_dangle_length:
        to_add.append(entry)


def nx_gx_identical(
    edges: gpd.GeoDataFrame,
    *,
    geom: shapely.Polygon,
    to_drop: list,
    to_add: list,
    nodes: gpd.GeoSeries,
    angle: float | int,
    max_segment_length: float | int = 1,
    clip_limit: float | int = 2,
    consolidation_tolerance: float | int = 10,
    eps: float = 1e-4,
) -> None:
    """Determine lines within artifacts to drop & add when dealing with typologies of
    more than 1 node and 1 or more identical continuity groups – ``{C, E, S}``.

    It is used when there are at least two nodes, one or more continuity groups but all
    edges have the same position in their respective continuity groups. For example,
    they all are ``E`` (ending), or ``S`` (single). It does not mean that all edges
    belong to a single continuity group. Here the "identical" refers to identical
    continuity groups – e.g. ``4CCC, ``5EE``, or ``3SSS``.

    Drop all of them and link the entry points to a centroid.

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Line geometries forming the artifact.
    geom : shapely.Polygon
        The polygonal representation of the artifact.
    to_drop : list
        List collecting geometries to be dropped.
    to_add : list
        List collecting geometries to be added.
    nodes : geopandas.GeoSeries
        Node geometries forming the artifact.
    angle : float | int
        Threshold for determination of line intersection angle acuteness.
        If the angle between two lines is too sharp, replace with a
        direct connection between the nodes.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
    clip_limit : float | int = 2
        Following generation of the Voronoi linework in ``geometry.voronoi_skeleton()``,
        we clip to fit inside the polygon. To ensure we get a space to make proper
        topological connections from the linework to the actual points on the edge of
        the polygon, we clip using a polygon with a negative buffer of ``clip_limit``
        or the radius of maximum inscribed circle, whichever is smaller.
    consolidation_tolerance : float | int = 10
         Tolerance passed to node consolidation within the
         ``geometry.voronoi_skeleton()``.
    eps : float = 1e-4
        Small tolerance epsilon.

    Returns
    -------
    None
        ``to_drop`` and ``to_add`` are updated inplace.
    """
    centroid = geom.centroid
    relevant_nodes = nodes.geometry.iloc[
        nodes.sindex.query(geom, predicate="dwithin", distance=eps)
    ]

    to_drop.extend(edges.index.to_list())
    lines = shapely.shortest_line(relevant_nodes, centroid)

    if not _is_within(lines, geom).all():
        logger.debug("NOT WITHIN replacing with skeleton")
        lines, _ = voronoi_skeleton(
            edges.geometry,  # use edges that are being dropped
            poly=geom,
            max_segment_length=max_segment_length,
            clip_limit=clip_limit,
            snap_to=relevant_nodes,
            consolidation_tolerance=consolidation_tolerance,
        )
        to_add.extend(weld_edges(lines, ignore=relevant_nodes.geometry))
    # if the angle between two lines is too sharp,
    # replace with a direct connection between the nodes
    elif len(lines) == 2:
        if angle_between_two_lines(lines.iloc[0], lines.iloc[1]) < angle:
            logger.debug(
                "TWO LINES WITH SHARP ANGLE replacing with straight connection"
            )
            to_add.append(
                shapely.LineString([relevant_nodes.iloc[0], relevant_nodes.iloc[1]])
            )
        else:
            to_add.extend(lines.tolist())
    else:
        to_add.extend(lines.tolist())


def nx_gx(
    edges: gpd.GeoDataFrame,
    *,
    artifact: gpd.GeoDataFrame,
    to_drop: list,
    to_add: list,
    split_points: list,
    nodes: gpd.GeoSeries,
    max_segment_length: float | int = 1,
    clip_limit: float | int = 2,
    min_dangle_length: float | int = 10,
    consolidation_tolerance: float | int = 10,
    eps: float = 1e-4,
) -> None:
    """Determine lines within artifacts to drop & add when dealing with typologies of
    2 or more nodes and 2 or more continuity groups – ``{C, E, S}``.

    Drop all but highest hierarchy. If there are unconnected nodes after drop, connect
    to nearest remaining edge or nearest intersection if there are more remaining edges.
    If there are three or more of the highest hierarchy, use the roundabout solution.

    If, after dropping, we end up with more than one connected component based on
    remaining edges, create a connection either as a shortest line between the two or
    using a skeleton if that is not inside or there are 3 or more components.

    The connection point should ideally be an existing
    nearest node with degree 4 or above.

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Line geometries forming the artifact.
    artifact : geopandas.GeoDataFrame
        The polygonal representation of the artifact.
    to_drop : list
        List collecting geometries to be dropped.
    to_add : list
        List collecting geometries to be added.
    split_points : list
        Points to be used for topological corrections.
    nodes : geopandas.GeoSeries
        Node geometries forming the artifact.
    clip_limit : float | int = 2
        Following generation of the Voronoi linework in ``geometry.voronoi_skeleton()``,
        we clip to fit inside the polygon. To ensure we get a space to make proper
        topological connections from the linework to the actual points on the edge of
        the polygon, we clip using a polygon with a negative buffer of ``clip_limit``
        or the radius of maximum inscribed circle, whichever is smaller.
    min_dangle_length : float | int = 10
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation within the
        ``geometry.voronoi_skeleton()``.
    eps : float = 1e-4
        Small tolerance epsilon.

    Returns
    -------
    None
        ``to_drop`` and ``to_add`` are updated inplace.
    """

    # filter ends
    all_ends = edges[edges.coins_end]

    # determine if we have C present or not. Based on that, ensure that we
    # correctly pick-up the highest hierarchy and drop all lower
    if artifact.C > 0:
        logger.debug("HIGHEST C")
    else:
        logger.debug("HIGHEST E")
        singles = set()
        visited = []
        for coins_count, group in zip(
            all_ends.coins_count, all_ends.coins_group, strict=True
        ):
            if (group not in visited) and (
                coins_count == (edges.coins_group == group).sum()
            ):
                singles.add(group)
                visited.append(group)
        # re-filter ends
        all_ends = edges[edges.coins_group.isin(singles)]

    # define mask for E and S strokes
    es_mask = edges.coins_group.isin(all_ends.coins_group)

    # filter Cs
    highest_hierarchy = edges[~es_mask]

    # get nodes forming the artifact
    relevant_nodes = nodes.iloc[
        nodes.sindex.query(artifact.geometry, predicate="dwithin", distance=eps)
    ]
    # filter nodes that lie on Cs (possibly primes)
    nodes_on_cont = relevant_nodes.index[
        relevant_nodes.sindex.query(
            highest_hierarchy.geometry.union_all(), predicate="dwithin", distance=eps
        )
    ]
    # get nodes that are not on Cs
    remaining_nodes = relevant_nodes.drop(nodes_on_cont)

    # get all remaining geometries and determine if they are all
    # connected or new connections need to happen
    remaining_geoms = pd.concat([remaining_nodes.geometry, highest_hierarchy.geometry])
    heads_ix, tails_ix = remaining_geoms.sindex.query(
        remaining_geoms, predicate="intersects"
    )
    n_comps = graph.Graph.from_arrays(heads_ix, tails_ix, 1).n_components

    # add list of existing edges to be removed from the network
    to_drop.extend(edges[es_mask].index.tolist())

    # more than one component in the remaining geometries
    # (either highest_hierarchy or remaining nodes)
    if n_comps > 1:
        logger.debug("CONDITION n_comps > 1 True")

        # get nodes that are relevant snapping targets (degree 4+)
        relevant_targets = relevant_nodes.loc[nodes_on_cont].query("degree > 3")

        cont_comp_labels = graph.Graph.build_contiguity(
            highest_hierarchy, rook=False
        ).component_labels
        conts_groups = highest_hierarchy.dissolve(cont_comp_labels)

        # BRANCH 1 - multiple Cs
        if len(highest_hierarchy) > 1:
            logger.debug("CONDITION len(highest_hierarchy) > 1 True")

            # Get nodes on Cs
            bd_points = highest_hierarchy.boundary.explode()
            # Identify nodes on primes
            primes = bd_points[bd_points.duplicated()]

            # For CCSS we need a special case solution if the length of S is
            # significantly shorter than the length of C. In that case, Voronoi does not
            # create shortest connections but a line that is parallel to Cs.
            if (
                highest_hierarchy.coins_group.nunique() == 2
                and artifact.S == 2
                and artifact.E == 0
                and (highest_hierarchy.length.sum() > all_ends.length.sum())
            ):
                logger.debug("CONDITION for CCSS special case True")

                # this also appends to split_points
                new_connections = ccss_special_case(
                    primes,
                    conts_groups,
                    highest_hierarchy,
                    relevant_nodes,
                    split_points,
                )

            else:
                logger.debug("CONDITION for CCSS special case False")

                # Get new connections via skeleton
                new_connections, splitters = voronoi_skeleton(
                    edges.geometry,  # use all edges as an input
                    poly=artifact.geometry,
                    snap_to=relevant_targets.geometry,  # snap to nodes
                    max_segment_length=max_segment_length,
                    clip_limit=clip_limit,
                    consolidation_tolerance=consolidation_tolerance,
                )

                # If there are multiple components, limit_distance was too drastic and
                # clipped the skeleton in pieces. Re-do it with a tiny epsilon.
                # This may cause tiny sharp angles but at least it will be connected.
                if (
                    graph.Graph.build_contiguity(
                        gpd.GeoSeries(new_connections), rook=False
                    ).n_components
                    > 1
                ):
                    # Get new connections via skeleton
                    new_connections, splitters = voronoi_skeleton(
                        edges.geometry,  # use all edges as an input
                        poly=artifact.geometry,
                        snap_to=relevant_targets.geometry,  # snap to nodes
                        max_segment_length=max_segment_length,
                        clip_limit=eps,
                        consolidation_tolerance=consolidation_tolerance,
                    )
                split_points.extend(splitters)

                # The skeleton returns connections to all the nodes. We need to keep
                # only some, if there are multiple connections to a single C. We don't
                # touch the other.

                (
                    new_connections,
                    connections_intersecting_c,
                    connections_intersecting_primes,
                ) = filter_connections(
                    primes, relevant_targets, conts_groups, new_connections
                )

                # mutliple Cs that are not intersecting. Avoid forks on the ends of
                # Voronoi. If one goes to relevant node, keep it. If not, remove both
                # and replace with a new shortest connection
                if (
                    len(connections_intersecting_c) > 1
                    and len(connections_intersecting_primes) == 0
                ):
                    # this also appends to split_points
                    new_connections = avoid_forks(
                        highest_hierarchy,
                        new_connections,
                        relevant_targets,
                        artifact,
                        split_points,
                    )

                # check for disconnected Cs and reconnect
                new_connections = reconnect(
                    conts_groups, new_connections, artifact, split_points, eps
                )

                # the drop above could've introduced a dangling edges. Remove those.
                new_connections = remove_dangles(new_connections, artifact)

        # BRANCH 2 - relevant node targets exist
        elif relevant_targets.shape[0] > 0:
            logger.debug("CONDITION relevant_targets.shape[0] > 0 True")

            # SUB BRANCH - only one remaining node
            if remaining_nodes.shape[0] < 2:
                logger.debug("CONDITION remaining_nodes.shape[0] < 2 True")

                # this also appends to split_points
                new_connections = one_remaining(
                    relevant_targets,
                    remaining_nodes,
                    artifact,
                    edges,
                    es_mask,
                    max_segment_length,
                    split_points,
                    clip_limit,
                    consolidation_tolerance,
                )

            # SUB BRANCH - more than one remaining node
            else:
                logger.debug("CONDITION remaining_nodes.shape[0] < 2 False")

                # this also appends to split_points
                new_connections = multiple_remaining(
                    edges,
                    es_mask,
                    artifact,
                    max_segment_length,
                    highest_hierarchy,
                    split_points,
                    relevant_targets.geometry,
                    clip_limit,
                    consolidation_tolerance,
                )

        # BRANCH 3 - no target nodes - snapping to C
        else:
            logger.debug("CONDITION relevant_targets.shape[0] > 0 False, snapping to C")

            # SUB BRANCH - only one remaining node
            if remaining_nodes.shape[0] < 2:
                logger.debug("CONDITION remaining_nodes.shape[0] < 2 True")

                # this also appends to split_points
                new_connections = one_remaining_c(
                    remaining_nodes,
                    highest_hierarchy,
                    artifact,
                    edges,
                    es_mask,
                    max_segment_length,
                    split_points,
                    clip_limit,
                    consolidation_tolerance,
                )

            # SUB BRANCH - more than one remaining node
            else:
                logger.debug("CONDITION remaining_nodes.shape[0] < 2 False")

                # this also appends to split_points
                new_connections = multiple_remaining(
                    edges,
                    es_mask,
                    artifact,
                    max_segment_length,
                    highest_hierarchy,
                    split_points,
                    highest_hierarchy.dissolve("coins_group").geometry,
                    clip_limit,
                    consolidation_tolerance,
                )

            new_connections = reconnect(
                conts_groups, new_connections, artifact, split_points, eps
            )

        # add new connections to a list of features to be added to the network
        to_add.extend(weld_edges(new_connections, ignore=remaining_nodes.geometry))

    # there may be loops or half-loops we are dropping. If they are protruding enough
    # we want to replace them by a dead-end representing their space
    elif artifact.C == 1 and (artifact.E + artifact.S) == 1:
        logger.debug("CONDITION is_loop True")

        sl = shapely.shortest_line(
            relevant_nodes.geometry.iloc[0], relevant_nodes.geometry.iloc[1]
        )

        if (
            (artifact.interstitial_nodes == 0)
            and _is_within(sl, artifact.geometry)
            and (sl.length * 1.1) < highest_hierarchy.length.sum()
        ):
            logger.debug("DEVIATION replacing with shortest")
            to_add.append(sl)
            to_drop.append(highest_hierarchy.index[0])

        else:
            dangles = loop(
                edges,
                es_mask,
                highest_hierarchy,
                artifact,
                max_segment_length,
                clip_limit,
                split_points,
                min_dangle_length,
            )
            if len(dangles) > 0:
                to_add.extend(weld_edges(dangles))

    elif artifact.node_count == 2 and artifact.stroke_count == 2:
        logger.debug("CONDITION is_sausage True")

        sl = shapely.shortest_line(
            relevant_nodes.geometry.iloc[0], relevant_nodes.geometry.iloc[1]
        )
        if (
            _is_within(sl, artifact.geometry)
            and (sl.length * 1.1) < highest_hierarchy.length.sum()
        ):
            logger.debug("DEVIATION replacing with shortest")
            to_add.append(sl)
            to_drop.append(highest_hierarchy.index[0])
    else:
        logger.debug("DROP ONLY")


def nx_gx_cluster(
    edges: gpd.GeoDataFrame,
    *,
    cluster_geom: gpd.GeoSeries,
    nodes: gpd.GeoSeries,
    to_drop: list,
    to_add: list,
    max_segment_length: float | int = 1,
    min_dangle_length: float | int = 20,
    consolidation_tolerance: float | int = 10,
    eps: float = 1e-4,
) -> None:
    """Determine lines within artifacts to drop & add when dealing with typologies of
    *clusters* of 2 or more nodes and 2 or more continuity groups – ``{C, E, S}``.

    Here :math:`n`-artifact cluster are treated as follows:
        * merge all artifact polygons;
        * drop all lines fully within the merged polygon;
        * skeletonize and keep only skeletonized edges and connecting nodes

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Line geometries forming the artifact.
    cluster_geom : geopandas.GeoSeries
        The polygonal representation of the artifact cluster.
    nodes : geopandas.GeoSeries
        Node geometries forming the artifact.
    to_drop : list
        List collecting geometries to be dropped.
    to_add : list
        List collecting geometries to be added.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
    min_dangle_length : float | int = 20
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    consolidation_tolerance : float | int = 10
         Tolerance passed to node consolidation within the
         ``geometry.voronoi_skeleton()``.
    eps : float = 1e-4
        Small tolerance epsilon.

    Returns
    -------
    None
        ``to_drop`` and ``to_add`` are updated inplace.
    """

    lines_to_drop = edges.iloc[
        edges.sindex.query(cluster_geom.buffer(eps), predicate="contains")
    ].index.to_list()
    connection = edges.drop(lines_to_drop).geometry
    # non-planar lines are not connections
    connection = connection[~connection.crosses(cluster_geom)]

    # if there's nothing to drop due to planarity, there's nothing to replace and
    # we can stop here
    if not lines_to_drop or connection.empty:
        return

    # get edges on boundary
    edges_on_boundary = edges.intersection(cluster_geom.boundary.buffer(eps)).explode(
        ignore_index=True
    )
    edges_on_boundary = edges_on_boundary[
        (~edges_on_boundary.is_empty)
        & (edges_on_boundary.geom_type.str.contains("Line"))
        & (edges_on_boundary.length > 100 * eps)
    ]  # keeping only (multi)linestrings of length>>eps
    edges_on_boundary = edges_on_boundary.to_frame("geometry")

    # find nodes ON the cluster polygon boundary (to be partially kept)
    nodes_on_boundary = nodes.iloc[
        nodes.sindex.query(cluster_geom.boundary.buffer(eps), predicate="intersects")
    ].copy()

    # find edges that cross but do not lie within
    edges_crossing = edges.iloc[
        edges.sindex.query(cluster_geom.buffer(eps), predicate="crosses")
    ]

    # the nodes to keep are those that intersect with these crossing edges
    nodes_to_keep = nodes_on_boundary.iloc[
        nodes_on_boundary.sindex.query(
            edges_crossing.union_all(), predicate="intersects"
        )
    ].copy()

    # merging lines between nodes to keep:
    buffered_nodes_to_keep = nodes_to_keep.buffer(eps).union_all()

    # make queen contiguity graph on MINUSBUFFERED outline road segments,
    # and copy component labels into edges_on_boundary gdf
    edges_on_boundary = edges_on_boundary.explode(ignore_index=True)
    queen = graph.Graph.build_fuzzy_contiguity(
        edges_on_boundary.difference(buffered_nodes_to_keep)
    )
    if len(connection) > 1:
        skeletonization_input = edges_on_boundary.dissolve(
            by=queen.component_labels
        ).geometry
    else:
        # a loop that has only a single entry point - use individual segments
        merged_edges = edges_on_boundary.dissolve().line_merge().item()
        if merged_edges.geom_type != "LineString":
            # this is a fallback for corner cases. It should result in the nearly the
            # same skeleton in the end but ensures we work with a single-part geometry
            merged_edges = shapely.concave_hull(merged_edges).exterior
        skeletonization_input = line_segments(merged_edges)

    # skeletonize
    skel, _ = voronoi_skeleton(
        skeletonization_input,
        cluster_geom,
        snap_to=False,
        max_segment_length=max_segment_length,
        clip_limit=1e-4,
        consolidation_tolerance=consolidation_tolerance,
    )

    # if we used only segments, we need to remove dangles
    if len(connection) == 1:
        connection = connection.item()
        _skel = gpd.GeoSeries(skel)
        _skel = _skel[
            _skel.disjoint(edges_on_boundary.union_all()) | _skel.intersects(connection)
        ]
        welded = gpd.GeoSeries(weld_edges(_skel))
        skel = welded[
            ~(
                ((welded.length < min_dangle_length) & (is_dangle(welded)))
                & welded.disjoint(connection)
            )
        ]

    lines_to_add = list(skel)

    to_drop.extend(lines_to_drop)

    ### RECONNECTING NON-PLANAR INTRUDING EDGES TO SKELETON

    # considering only edges that are kept
    edges_kept = edges.copy().drop(lines_to_drop, axis=0)

    to_reconnect = []

    skel_merged = shapely.line_merge(skel)
    skel_merged = gpd.GeoSeries(skel_merged, crs=edges.crs)

    skel_nodes = list(shapely.get_point(skel_merged, 0))
    skel_nodes.extend(list(shapely.get_point(skel_merged, -1)))
    skel_nodes = gpd.GeoSeries(skel_nodes, crs=edges.crs).union_all()

    # loop through endpoints of kept edges...
    for i in [0, -1]:
        # do the same for "end" points
        endpoints = gpd.GeoSeries(
            shapely.get_point(edges_kept.geometry, i), crs=edges.crs
        )

        # which are contained by artifact...
        endpoints = endpoints.iloc[
            endpoints.sindex.query(cluster_geom, predicate="contains")
        ]

        # ...but NOT on skeleton
        endpoints = endpoints.difference(skel_merged.union_all())

        to_reconnect.extend(endpoints.geometry.drop_duplicates())

    # to_reconnect now contains a list of points which need to be connected to the
    # nearest skel node: from those nodes, we need to add shapely shortest lines between
    # those edges_kept.endpoints and
    non_planar_connections = shapely.shortest_line(skel_nodes, to_reconnect)

    # keep only those that are within
    conn_within = _is_within(non_planar_connections, cluster_geom)
    if not all(conn_within):
        warnings.warn(
            "Could not create a connection as it would lead outside of the artifact.",
            UserWarning,
            stacklevel=2,
        )
    non_planar_connections = non_planar_connections[conn_within]

    ### extend our list "to_add" with this artifact clusters' contribution:
    lines_to_add.extend(non_planar_connections)
    to_add.extend(lines_to_add)


def is_dangle(edgelines: gpd.GeoSeries) -> bool:
    """Determine if an edge is dangling or not."""

    def _sum_intersects(loc: int) -> int:
        """Sum the number of places linestrings intersect each other."""
        point = shapely.get_point(edgelines, loc)
        ix, edge_ix1 = edgelines.sindex.query(point, predicate="intersects")
        data = ([True] * len(ix), (ix, edge_ix1))
        return sparse.coo_array(data, shape=shape, dtype=np.bool_).sum(axis=1)

    shape = (len(edgelines), len(edgelines))
    first_sum = _sum_intersects(0)
    last_sum = _sum_intersects(-1)

    return (first_sum == 1) | (last_sum == 1)


def line_segments(line: shapely.LineString) -> np.ndarray:
    """Explode a linestring into constituent pairwise coordinates."""
    xys = shapely.get_coordinates(line)
    return shapely.linestrings(
        np.column_stack((xys[:-1], xys[1:])).reshape(xys.shape[0] - 1, 2, 2)
    )
