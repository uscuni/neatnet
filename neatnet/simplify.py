import logging
import typing
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from geopandas.testing import assert_geoseries_equal
from libpysal import graph
from scipy import sparse

from .artifacts import (
    get_artifacts,
    n1_g1_identical,
    nx_gx,
    nx_gx_cluster,
    nx_gx_identical,
)
from .continuity import continuity, get_stroke_info
from .nodes import (
    _nodes_degrees_from_edges,
    _nodes_from_edges,
    _status,
    consolidate_nodes,
    fix_topology,
    induce_nodes,
    remove_interstitial_nodes,
    split,
)

DEBUGGING = False

logger = logging.getLogger(__name__)


def _check_input_crs(streets: gpd.GeoDataFrame, exclusion_mask: gpd.GeoSeries):
    """Ensure input data is in appropriate Coordinate reference systems."""

    streets_crs = streets.crs
    streets_has_crs = streets_crs is not None

    if not streets_has_crs:
        warnings.warn(
            (
                "The input `streets` data does not have an assigned "
                "coordinate reference system. Assuming a projected CRS in meters."
            ),
            category=UserWarning,
            stacklevel=2,
        )

    else:
        if not streets_crs.is_projected:
            raise ValueError(
                "The input `streets` data are not in a projected "
                "coordinate reference system. Reproject and rerun."
            )

        if streets_crs.axis_info[0].unit_name != "metre":
            warnings.warn(
                (
                    "The input `streets` data coordinate reference system is projected "
                    "but not in meters. All `neatnet` defaults assume meters. "
                    "Either reproject and rerun or proceed with caution."
                ),
                category=UserWarning,
                stacklevel=2,
            )

    if exclusion_mask is not None and exclusion_mask.crs != streets_crs:
        raise ValueError(
            "The input `streets` and `exclusion_mask` data are in "
            "different coordinate reference systems. Reproject and rerun."
        )


def _link_nodes_artifacts(
    step: str,
    streets: gpd.GeoDataFrame,
    artifacts: gpd.GeoDataFrame,
    eps: None | float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Helper to prep nodes & artifacts when simplifying singletons & pairs."""

    # Get nodes from the network
    nodes = _nodes_degrees_from_edges(streets.geometry)

    if step == "singletons":
        node_geom = nodes.geometry
        sindex_kwargs = {"predicate": "dwithin", "distance": eps}
    else:
        node_geom = nodes.buffer(0.1)
        sindex_kwargs = {"predicate": "intersects"}

    # Link nodes to artifacts
    node_idx, artifact_idx = artifacts.sindex.query(node_geom, **sindex_kwargs)

    intersects = sparse.coo_array(
        ([True] * len(node_idx), (node_idx, artifact_idx)),
        shape=(len(nodes), len(artifacts)),
        dtype=np.bool_,
    )

    # Compute number of nodes per artifact
    artifacts["node_count"] = intersects.sum(axis=0)

    return nodes, artifacts


def _classify_strokes(
    artifacts: gpd.GeoDataFrame, streets: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Classify artifacts with ``{C,E,S}`` typology."""

    strokes, c_, e_, s_ = get_stroke_info(artifacts, streets)

    artifacts["stroke_count"] = strokes
    artifacts["C"] = c_
    artifacts["E"] = e_
    artifacts["S"] = s_

    return artifacts


def _identify_non_planar(
    artifacts: gpd.GeoDataFrame, streets: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Filter artifacts caused by non-planar intersections."""

    # Note from within `neatify_singletons()`
    # TODO: This is not perfect.
    # TODO: Some 3CC artifacts were non-planar but not captured here.

    artifacts["non_planar"] = artifacts["stroke_count"] > artifacts["node_count"]
    a_idx, r_idx = streets.sindex.query(
        artifacts.geometry.boundary, predicate="overlaps"
    )
    artifacts.iloc[np.unique(a_idx), artifacts.columns.get_loc("non_planar")] = True

    return artifacts


def neatify_singletons(
    artifacts: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    *,
    max_segment_length: float | int = 1,
    compute_coins: bool = True,
    angle_threshold: float = 120,
    min_dangle_length: float | int = 10,
    eps: float = 1e-4,
    clip_limit: float | int = 2,
    simplification_factor: float | int = 2,
    consolidation_tolerance: float | int = 10,
) -> gpd.GeoDataFrame:
    """Simplification of singleton face artifacts – the first simplification step in
    the procedure detailed in ``simplify.neatify_loop()``.

    This process extracts nodes from network edges before computing and labeling
    face artifacts with a ``{C, E, S}`` typology through ``momepy.COINS`` via the
    constituent street geometries.

    Next, each artifact is iterated over and constituent line geometries are either
    dropped or added in the following order of typologies:

        1. 1 node and 1 continuity group
        2. more than 1 node and 1 or more identical continuity groups
        3. 2 or more nodes and 2 or more continuity groups

    Non-planar geometries are ignored.

    Parameters
    ----------
    artifacts : geopandas.GeoDataFrame
        Face artifact polygons.
    streets : geopandas.GeoDataFrame
        Preprocessed street network data.
    max_segment_length : float | int = 1
        Additional nodes will be added so that all line segments
        are no longer than this value. Must be greater than 0.
        Used in multiple internal geometric operations.
    compute_coins : bool = True
        Flag for computing and labeling artifacts with a ``{C, E, S}`` typology through
        :class:`momepy.COINS` via the constituent street geometries.
    angle_threshold : float = 120
        See the ``angle_threshold`` keyword argument in :class:`momepy.COINS`.
    min_dangle_length : float | int = 10
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    eps : float = 1e-4
        Tolerance epsilon used in multiple internal geometric operations.
    clip_limit : float | int = 2
        Following generation of the Voronoi linework, we clip to fit inside the
        polygon. To ensure we get a space to make proper topological connections
        from the linework to the actual points on the edge of the polygon, we clip
        using a polygon with a negative buffer of ``clip_limit`` or the radius of
        maximum inscribed circle, whichever is smaller.
    simplification_factor : float | int = 2
        The factor by which singles, pairs, and clusters are simplified. The
        ``max_segment_length`` is multiplied by this factor to get the
        simplification epsilon.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation when generating Voronoi skeletons.

    Returns
    -------
    geopandas.GeoDataFrame
        The street network line data following the singleton procedure.
    """

    # Extract network nodes and relate to artifacts
    nodes, artifacts = _link_nodes_artifacts("singletons", streets, artifacts, eps)

    # Compute number of stroke groups per artifact
    if compute_coins:
        streets, _ = continuity(streets, angle_threshold=angle_threshold)
    artifacts = _classify_strokes(artifacts, streets)

    # Filter artifacts caused by non-planar intersections
    artifacts = _identify_non_planar(artifacts, streets)

    # Count intersititial nodes (primes)
    _prime_count = artifacts["node_count"] - artifacts[["C", "E", "S"]].sum(axis=1)
    artifacts["interstitial_nodes"] = _prime_count

    # Define the type label
    ces_type = []
    for x in artifacts[["node_count", "C", "E", "S"]].itertuples():
        ces_type.append(f"{x.node_count}{'C' * x.C}{'E' * x.E}{'S' * x.S}")
    artifacts["ces_type"] = ces_type

    # Collect changes
    to_drop: list[int] = []
    to_add: list[int] = []
    split_points: list[shapely.Point] = []

    # Isolate planar artifacts
    planar = artifacts[~artifacts["non_planar"]].copy()
    planar["buffered"] = planar.buffer(eps)
    if artifacts["non_planar"].any():
        logger.debug(f"IGNORING {artifacts.non_planar.sum()} non planar artifacts")

    # Iterate over each singleton planar artifact and simplify based on typology
    for artifact in planar.itertuples():
        n_nodes = artifact.node_count
        n_strokes = artifact.stroke_count
        cestype = artifact.ces_type

        # Get edges relevant for an artifact
        edges = streets.iloc[
            streets.sindex.query(artifact.buffered, predicate="covers")
        ]

        # Dispatch by typology
        try:
            # 1 node and 1 continuity group
            if (n_nodes == 1) and (n_strokes == 1):
                logger.debug("FUNCTION n1_g1_identical")
                n1_g1_identical(
                    edges,
                    to_drop=to_drop,
                    to_add=to_add,
                    geom=artifact.geometry,
                    max_segment_length=max_segment_length,
                    clip_limit=clip_limit,
                )
            # More than 1 node and 1 or more identical continuity groups
            elif (n_nodes > 1) and (len(set(cestype[1:])) == 1):
                logger.debug("FUNCTION nx_gx_identical")
                nx_gx_identical(
                    edges,
                    geom=artifact.geometry,
                    to_add=to_add,
                    to_drop=to_drop,
                    nodes=nodes,
                    angle=75,
                    max_segment_length=max_segment_length,
                    clip_limit=clip_limit,
                    consolidation_tolerance=consolidation_tolerance,
                )
            # 2 or more nodes and 2 or more continuity groups
            elif (n_nodes > 1) and (len(cestype) > 2):
                logger.debug("FUNCTION nx_gx")
                nx_gx(
                    edges,
                    artifact=artifact,
                    to_drop=to_drop,
                    to_add=to_add,
                    split_points=split_points,
                    nodes=nodes,
                    max_segment_length=max_segment_length,
                    clip_limit=clip_limit,
                    min_dangle_length=min_dangle_length,
                    consolidation_tolerance=consolidation_tolerance,
                )
            else:
                logger.debug("NON PLANAR")
        except Exception as e:
            if DEBUGGING:
                raise e
            warnings.warn(
                f"An error occured at location {artifact.geometry.centroid}. "
                f"The artifact has not been simplified. The original message:\n{e}",
                UserWarning,
                stacklevel=2,
            )

    cleaned_streets = streets.drop(to_drop)
    # split lines on new nodes
    cleaned_streets = split(split_points, streets.drop(to_drop), streets.crs)

    if to_add:
        # Create new streets with fixed geometry.
        # Note: ``to_add`` and ``to_drop`` lists shall be global and
        # this step should happen only once, not for every artifact
        _add_merged = gpd.GeoSeries(to_add).line_merge()
        new = gpd.GeoDataFrame(geometry=_add_merged, crs=streets.crs).explode()
        new = new[~new.normalize().duplicated()].copy()
        new["_status"] = "new"
        new.geometry = new.simplify(max_segment_length * simplification_factor)
        new_streets = pd.concat([cleaned_streets, new], ignore_index=True)
        agg: dict[str, str | typing.Callable] = {"_status": _status}
        for c in cleaned_streets.columns.drop(cleaned_streets.active_geometry_name):
            if c != "_status":
                agg[c] = "first"
        non_empties = new_streets[~(new_streets.is_empty | new_streets.geometry.isna())]
        new_streets = remove_interstitial_nodes(non_empties, aggfunc=agg)

        final = new_streets
    else:
        final = cleaned_streets

    if "coins_group" in final.columns:
        final = final.drop(
            columns=[c for c in streets.columns if c.startswith("coins_")]
        )
    return final


def neatify_pairs(
    artifacts: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    *,
    max_segment_length: float | int = 1,
    min_dangle_length: float | int = 20,
    clip_limit: float | int = 2,
    simplification_factor: float | int = 2,
    consolidation_tolerance: float | int = 10,
) -> gpd.GeoDataFrame:
    """Simplification of pairs of face artifacts – the second simplification step in
    the procedure detailed in ``simplify.neatify_loop()``.

    This process extracts nodes from network edges before identifying non-planarity
    and cluster information.

    If paired artifacts are present we further classify them as grouped by
    first vs. last instance of duplicated component label, and whether
    or not to be simplified with the clustered process.

    Finally, simplification is performed based on the following order of typologies:
        1. Singletons – merged pairs & first instance (w/o COINS)
        2. Singletons – Second instance – w/ COINS
        3. Clusters

    Parameters
    ----------
    artifacts : geopandas.GeoDataFrame
        Face artifact polygons.
    streets : geopandas.GeoDataFrame
        Preprocessed street network data.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
        Used in multiple internal geometric operations.
    min_dangle_length : float | int = 20
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    clip_limit : float | int = 2
        Following generation of the Voronoi linework, we clip to fit inside the
        polygon. To ensure we get a space to make proper topological connections
        from the linework to the actual points on the edge of the polygon, we clip
        using a polygon with a negative buffer of ``clip_limit`` or the radius of
        maximum inscribed circle, whichever is smaller.
    simplification_factor : float | int = 2
        The factor by which singles, pairs, and clusters are simplified. The
        ``max_segment_length`` is multiplied by this factor to get the
        simplification epsilon.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation when generating Voronoi skeletons.

    Returns
    -------
    geopandas.GeoDataFrame
        The street network line data following the pairs procedure.
    """

    # Extract network nodes and relate to artifacts
    nodes, artifacts = _link_nodes_artifacts("pairs", streets, artifacts, None)

    # Compute number of stroke groups per artifact
    streets, _ = continuity(streets)
    artifacts = _classify_strokes(artifacts, streets)

    # Filter artifacts caused by non-planar intersections
    artifacts = _identify_non_planar(artifacts, streets)

    # Identify non-planar clusters
    _id_np = lambda x: sum(artifacts.loc[artifacts["comp"] == x.comp]["non_planar"])  # noqa: E731
    artifacts["non_planar_cluster"] = artifacts.apply(_id_np, axis=1)
    # Subset non-planar clusters and planar artifacts
    np_clusters = artifacts[artifacts.non_planar_cluster > 0]
    artifacts_planar = artifacts[artifacts.non_planar_cluster == 0]

    # Isolate planar artifacts
    _planar_grouped = artifacts_planar.groupby("comp")[artifacts_planar.columns]
    _solutions = _planar_grouped.apply(get_solution, streets=streets)
    artifacts_w_info = artifacts.merge(_solutions, left_on="comp", right_index=True)

    # Isolate non-planar clusters of value 2 – e.g., artifact under highway
    _np_clust_2 = np_clusters["non_planar_cluster"] == 2
    artifacts_under_np = np_clusters[_np_clust_2].dissolve("comp", as_index=False)

    # Determine typology dispatch if artifacts are present
    if not artifacts_w_info.empty:
        agg = {
            "coins_group": "first",
            "coins_end": lambda x: x.any(),
            "_status": _status,
        }
        for c in streets.columns.drop(
            [streets.active_geometry_name, "coins_count"], errors="ignore"
        ):
            if c not in agg:
                agg[c] = "first"

        sol_drop = "solution == 'drop_interline'"
        sol_iter = "solution == 'iterate'"

        # Determine artifacts and street edges to drop
        _to_drop = artifacts_w_info.drop_duplicates("comp").query(sol_drop).drop_id
        _drop_streets = streets.drop(_to_drop.dropna().values)

        # Re-run node cleaning on subset of fresh street edges
        streets_cleaned = remove_interstitial_nodes(
            _drop_streets,
            aggfunc=agg,
        )

        # Isolate drops to create merged pairs
        merged_pairs = artifacts_w_info.query(sol_drop).dissolve("comp", as_index=False)

        # Sort artifacts by their node count low-to-high
        sorted_node_count = artifacts_w_info.sort_values("node_count", ascending=False)

        # Isolate artifacts to process as singletons – first instance
        _1st = sorted_node_count.query(sol_iter).drop_duplicates("comp", keep="first")
        _planar_clusters = np_clusters[~np_clusters["non_planar"]]
        _1st = pd.concat([_1st, _planar_clusters], ignore_index=True)

        # Isolate artifacts to process as singletons – last instance
        _2nd = sorted_node_count.query(sol_iter).drop_duplicates("comp", keep="last")

        # Isolate artifacts to process as clusters
        for_skeleton = artifacts_w_info.query("solution == 'skeleton'")

    # Otherwise instantiate artifact containers as empty
    else:
        merged_pairs = pd.DataFrame()
        _1st = pd.DataFrame()
        _2nd = pd.DataFrame()
        for_skeleton = pd.DataFrame()
        streets_cleaned = streets

    # Generate counts of COINs groups for edges
    coins_count = (
        streets_cleaned.groupby("coins_group", as_index=False)
        .geometry.count()
        .rename(columns={"geometry": "coins_count"})
    )
    streets_cleaned = streets_cleaned.merge(coins_count, on="coins_group", how="left")

    # Add under non-planars to cluster dispatcher
    if not artifacts_under_np.empty:
        for_skeleton = pd.concat([for_skeleton, artifacts_under_np])

    # Dispatch singleton simplifier
    if not merged_pairs.empty or not _1st.empty:
        # Merged pairs & first instance – w/o COINS
        streets_cleaned = neatify_singletons(
            pd.concat([merged_pairs, _1st]),
            streets_cleaned,
            max_segment_length=max_segment_length,
            clip_limit=clip_limit,
            compute_coins=False,
            min_dangle_length=min_dangle_length,
            simplification_factor=simplification_factor,
            consolidation_tolerance=consolidation_tolerance,
        )
        # Second instance – w/ COINS
        if not _2nd.empty:
            streets_cleaned = neatify_singletons(
                _2nd,
                streets_cleaned,
                max_segment_length=max_segment_length,
                clip_limit=clip_limit,
                compute_coins=True,
                min_dangle_length=min_dangle_length,
                simplification_factor=simplification_factor,
                consolidation_tolerance=consolidation_tolerance,
            )

    # Dispatch cluster simplifier
    if not for_skeleton.empty:
        streets_cleaned = neatify_clusters(
            for_skeleton,
            streets_cleaned,
            max_segment_length=max_segment_length,
            simplification_factor=simplification_factor,
            min_dangle_length=min_dangle_length,
            consolidation_tolerance=consolidation_tolerance,
        )

    return streets_cleaned


def neatify_clusters(
    artifacts: gpd.GeoDataFrame,
    streets: gpd.GeoDataFrame,
    *,
    max_segment_length: float | int = 1,
    eps: float = 1e-4,
    simplification_factor: float | int = 2,
    min_dangle_length: float | int = 20,
    consolidation_tolerance: float | int = 10,
) -> gpd.GeoDataFrame:
    """Simplification of clusters of face artifacts – the third simplification step in
    the procedure detailed in ``simplify.neatify_loop()``.

    This process extracts nodes from network edges before iterating over each
    cluster artifact and performing simplification.

    Parameters
    ----------
    artifacts : geopandas.GeoDataFrame
        Face artifact polygons.
    streets : geopandas.GeoDataFrame
        Preprocessed street network data.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
        Used in multiple internal geometric operations.
    eps : float = 1e-4
        Tolerance epsilon used in multiple internal geometric operations.
    simplification_factor : float | int = 2
        The factor by which singles, pairs, and clusters are simplified. The
        ``max_segment_length`` is multiplied by this factor to get the
        simplification epsilon.
    min_dangle_length : float | int = 20
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation when generating Voronoi skeletons.

    Returns
    -------
    geopandas.GeoDataFrame
        The street network line data following the clusters procedure.
    """

    # Get nodes from the network
    nodes = gpd.GeoSeries(_nodes_from_edges(streets.geometry))

    # Collect changes
    to_drop: list[int] = []
    to_add: list[int] = []

    for _, artifact in artifacts.groupby("comp"):
        # Get artifact cluster polygon
        cluster_geom = artifact.union_all()
        # Get edges relevant for an artifact
        edges = streets.iloc[
            streets.sindex.query(cluster_geom, predicate="intersects")
        ].copy()

        # Clusters of 2 or more nodes and 2 or more continuity groups
        nx_gx_cluster(
            edges=edges,
            cluster_geom=cluster_geom,
            nodes=nodes,
            to_drop=to_drop,
            to_add=to_add,
            eps=eps,
            max_segment_length=max_segment_length,
            min_dangle_length=min_dangle_length,
            consolidation_tolerance=consolidation_tolerance,
        )

    cleaned_streets = streets.drop(to_drop)

    # Create new street with fixed geometry.
    # Note: ``to_add`` and ``to_drop`` lists shall be global and
    # this step should happen only once, not for every artifact
    new = gpd.GeoDataFrame(geometry=to_add, crs=streets.crs)
    new["_status"] = "new"
    new["geometry"] = new.line_merge().simplify(
        max_segment_length * simplification_factor
    )
    new_streets = pd.concat([cleaned_streets, new], ignore_index=True).explode()
    agg: dict[str, str | typing.Callable] = {"_status": _status}
    for c in new_streets.columns.drop(new_streets.active_geometry_name):
        if c != "_status":
            agg[c] = "first"
    new_streets = remove_interstitial_nodes(
        new_streets[~new_streets.is_empty], aggfunc=agg
    ).drop_duplicates("geometry")

    return new_streets


def get_type(edges: gpd.GeoDataFrame, shared_edge: int) -> str:
    """Classify artifact edges according to the ``{C, E, S}``
    schema when considering solutions for pairs of artifacts.

    Parameters
    ----------
    edges : geopandas.GeoDataFrame
        Artifact edges in consideration.
    shared_edge : int
        The index location of the shared edge of the pair.

    Returns
    -------
    str
        Classification for an edge in ``{C, E, S}``.
    """

    if (  # Roundabout special case
        edges["coins_group"].nunique() == 1
        and edges.shape[0] == edges["coins_count"].iloc[0]
    ):
        return "S"

    all_ends = edges[edges["coins_end"]]
    mains = edges[~edges["coins_group"].isin(all_ends["coins_group"])]
    shared = edges.loc[shared_edge]

    if shared_edge in mains.index:
        return "C"

    if shared["coins_count"] == (edges["coins_group"] == shared["coins_group"]).sum():
        return "S"

    return "E"


def get_solution(group: gpd.GeoDataFrame, streets: gpd.GeoDataFrame) -> pd.Series:
    """Determine the solution for paired planar artifacts.

    Parameters
    ----------
    group : geopandas.GeoDataFrame
        Dissolved group of connected planar artifacts.
    streets : geopandas.GeoDataFrame
        Street network data.

    Returns
    -------
    pandas.Series
        The determined solution and edge to drop.
    """

    def _relate(loc: int) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Isolate intersecting & covering street geometries."""
        _geom = group.geometry.iloc[loc]
        _streets = streets.iloc[streets.sindex.query(_geom, predicate="intersects")]
        _covers = _streets.iloc[_streets.sindex.query(_geom, predicate="covers")]
        return _streets, _covers

    cluster_geom = group.union_all()

    streets_a, covers_a = _relate(0)
    streets_b, covers_b = _relate(1)

    # Find the street segment that is contained within the cluster geometry
    shared = streets.index[streets.sindex.query(cluster_geom, predicate="contains")]

    if shared.empty or covers_a.empty or covers_b.empty:
        return pd.Series({"solution": "non_planar", "drop_id": None})

    shared = shared.item()

    if (np.invert(streets_b.index.isin(covers_a.index)).sum() == 1) or (
        np.invert(streets_a.index.isin(covers_b.index)).sum() == 1
    ):
        return pd.Series({"solution": "drop_interline", "drop_id": shared})

    seen_by_a = get_type(covers_a, shared)
    seen_by_b = get_type(covers_b, shared)

    if seen_by_a == "C" and seen_by_b == "C":
        return pd.Series({"solution": "iterate", "drop_id": shared})

    if seen_by_a == seen_by_b:
        return pd.Series({"solution": "drop_interline", "drop_id": shared})

    return pd.Series({"solution": "skeleton", "drop_id": shared})


def neatify(
    streets: gpd.GeoDataFrame,
    *,
    exclusion_mask: None | gpd.GeoSeries = None,
    predicate: str = "intersects",
    max_segment_length: float | int = 1,
    min_dangle_length: float | int = 20,
    clip_limit: float | int = 2,
    simplification_factor: float | int = 2,
    consolidation_tolerance: float | int = 10,
    artifact_threshold: None | float | int = None,
    artifact_threshold_fallback: float | int = 7,
    area_threshold_blocks: float | int = 1e5,
    isoareal_threshold_blocks: float | int = 0.5,
    area_threshold_circles: float | int = 5e4,
    isoareal_threshold_circles_enclosed: float | int = 0.75,
    isoperimetric_threshold_circles_touching: float | int = 0.9,
    angle_threshold: float = 120,
    eps: float = 1e-4,
    n_loops: int = 2,
) -> gpd.GeoDataFrame:
    """Top-level workflow for simplifying street networks. The input raw street network
    data, which must be in a projected coordinate reference system and is expected to be
    in meters, is first preprocessed (topological corrections & node consolidation)
    before two iterations of artifact detection and simplification.

    Each iteration of the simplification procedure which includes (1.) the removal
    of false nodes; (2.) face artifact classification; and (3.) the line-based
    simplification of face artifacts in the order of single artifacts, pairs of
    artifacts, clusters of artifacts.

    For further information on face artifact detection and extraction
    see :cite:`fleischmann_shape-based_2024`.

    This algorithm is designed for use with only "street" network geometries as input.
    While passing in other types of pathing (e.g., sidewalks, canals) will likely yield
    valid geometric results, that behavior is untested.

    Parameters
    ----------
    streets : geopandas.GeoDataFrame
        Raw street network data. This input *must* be in a projected coordinate
        reference system and *should* be in meters. All defaults arguments assume
        meters. The internal algorithm is designed for use with street network
        geometries, not  other types of pathing (e.g., sidewalks, canals), which
        should be filtered out.
    exclusion_mask : None | geopandas.GeoSeries = None
        Geometries used to determine face artifacts to exclude from returned output.
    predicate : str = 'intersects'
        The spatial predicate used to exclude face artifacts from returned output.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
        Used in multiple internal geometric operations.
    min_dangle_length : float | int
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    clip_limit : float | int = 2
        Following generation of the Voronoi linework, we clip to fit inside the
        polygon. To ensure we get a space to make proper topological connections
        from the linework to the actual points on the edge of the polygon, we clip
        using a polygon with a negative buffer of ``clip_limit`` or the radius of
        maximum inscribed circle, whichever is smaller.
    simplification_factor : float | int = 2
        The factor by which singles, pairs, and clusters are simplified. The
        ``max_segment_length`` is multiplied by this factor to get the
        simplification epsilon.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation when generating Voronoi skeletons.
    artifact_threshold : None | float | int = None
        When ``artifact_threshold`` is passed, the computed value from
        ``momepy.FaceArtifacts.threshold`` is not used in favor of the
        given value. This is useful for small networks where artifact
        detection may fail or become unreliable.
    artifact_threshold_fallback : float | int = 7
        If artifact threshold detection fails, this value is used as a fallback.
    area_threshold_blocks : float | int = 1e5
        This is the first threshold for detecting block-like artifacts whose
        Face Artifact Index (see :cite:`fleischmann_shape-based_2024`) is above
        the value passed in ``artifact_threshold``.
        If a polygon has an area below ``area_threshold_blocks``, *and*
        is of elongated shape (see also ``isoareal_threshold_blocks``),
        *and* touches at least one polygon that has already been classified as artifact,
        then it will be classified as an artifact.
    isoareal_threshold_blocks : float | int = 0.5
        This is the second threshold for detecting block-like artifacts whose
        Face Artifact Index (see :cite:`fleischmann_shape-based_2024`) is above the
        value passed in ``artifact_threshold``. If a polygon has an isoareal quotient
        below ``isoareal_threshold_blocks`` (see ``esda.shape.isoareal_quotient``),
        i.e., if it has an elongated shape; *and* it has a sufficiently small area
        (see also ``area_threshold_blocks``), *and* if it touches at least one
        polygon that has already been detected as an artifact,
        then it will be classified as an artifact.
    area_threshold_circles : float | int = 5e4
        This is the first threshold for detecting circle-like artifacts whose
        Face Artifact Index (see :cite:`fleischmann_shape-based_2024`) is above the
        value passed in ``artifact_threshold``. If a polygon has an area below
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
        Face Artifact Index (see :cite:`fleischmann_shape-based_2024`) is above the
        value  passed in ``artifact_threshold``. If a polygon has a sufficiently small
        area (see also ``area_threshold_circles``), *and* the polygon is
        fully enclosed by polygons already classified as artifacts,
        *and* its isoareal quotient (see ``esda.shape.isoareal_quotient``)
        is above the value passed to ``isoareal_threshold_circles_enclosed``,
        i.e., if its shape is close to circular;
        then it will be classified as an artifact.
    isoperimetric_threshold_circles_touching : float | int = 0.9
        This is the third threshold for detecting circle-like artifacts whose
        Face Artifact Index (see :cite:`fleischmann_shape-based_2024`)
        is above the value passed in ``artifact_threshold``.
        If a polygon has a sufficiently small area
        (see also ``area_threshold_circles``), *and* the polygon is touched
        by at least one polygon already classified as artifact,
        *and* its isoperimetric quotient (see ``esda.shape.isoperimetric_quotient``)
        is above the value passed to ``isoperimetric_threshold_circles_touching``,
        i.e., if its shape is close to circular;
        then it will be classified as an artifact.
    angle_threshold : float = 120
        See the ``angle_threshold`` keyword argument in :class:`momepy.COINS`.
    eps : float = 1e-4
        Tolerance epsilon used in multiple internal geometric operations.
    n_loops : int = 2
        Number of loops through the simplification pipeline. It is recommended to stick
        to the default value and increase it only very conservatively.

    Returns
    -------
    geopandas.GeoDataFrame
        The final, simplified street network line data.

    Notes
    -----
    As is noted above, the input network data must be in a projected coordinate
    reference system and is expected to be in meters. However, it may be possible to
    work with network data projected in feet if all default arguments are adjusted.
    """

    _check_input_crs(streets, exclusion_mask)

    # Record state of initial input to compare with results from
    # -- topo fix & node consolidation if there are no artifacts
    raw_streets = streets.copy()

    # NOTE: this keeps attributes but resets index
    streets = fix_topology(streets, eps=eps)

    # Merge nearby nodes (up to double of distance used in skeleton).
    streets = consolidate_nodes(streets, tolerance=max_segment_length * 2.1)

    # Identify artifacts
    artifacts, threshold = get_artifacts(
        streets,
        exclusion_mask=exclusion_mask,
        predicate=predicate,
        threshold=artifact_threshold,
        threshold_fallback=artifact_threshold_fallback,
        area_threshold_blocks=area_threshold_blocks,
        isoareal_threshold_blocks=isoareal_threshold_blocks,
        area_threshold_circles=area_threshold_circles,
        isoareal_threshold_circles_enclosed=isoareal_threshold_circles_enclosed,
        isoperimetric_threshold_circles_touching=isoperimetric_threshold_circles_touching,
    )

    # If no artifacts return either the raw streets or topologically-fixed streets
    if artifacts.empty:
        try:
            assert_geoseries_equal(streets.geometry, raw_streets.geometry)
            warnings.warn(
                (
                    "No topological corrections performed on input `streets` "
                    "and no artifacts were detected. Returning as is."
                ),
                UserWarning,
                stacklevel=2,
            )
            return raw_streets
        except AssertionError:
            warnings.warn(
                (
                    "Topological corrections performed on input `streets` "
                    "but no artifacts were detected. Returning the results of "
                    "`fix_topology()` and `consolidate_nodes()`."
                ),
                UserWarning,
                stacklevel=2,
            )
            return streets

    # Loop 1
    new_streets = neatify_loop(
        streets,
        artifacts,
        max_segment_length=max_segment_length,
        min_dangle_length=min_dangle_length,
        clip_limit=clip_limit,
        simplification_factor=simplification_factor,
        consolidation_tolerance=consolidation_tolerance,
        eps=eps,
        angle_threshold=angle_threshold,
    )

    # This is potentially fixing some minor erroneous edges coming from Voronoi
    new_streets = induce_nodes(new_streets, eps=eps)
    new_streets = new_streets[~new_streets.geometry.normalize().duplicated()].copy()

    for _ in range(2, n_loops + 1):
        # Identify artifacts based on the first loop network
        artifacts, _ = get_artifacts(
            new_streets,
            threshold=threshold,
            threshold_fallback=artifact_threshold_fallback,
            area_threshold_blocks=area_threshold_blocks,
            isoareal_threshold_blocks=isoareal_threshold_blocks,
            area_threshold_circles=area_threshold_circles,
            isoareal_threshold_circles_enclosed=isoareal_threshold_circles_enclosed,
            isoperimetric_threshold_circles_touching=isoperimetric_threshold_circles_touching,
            exclusion_mask=exclusion_mask,
            predicate=predicate,
        )

        if artifacts.empty:
            return new_streets.reset_index(drop=True)

        new_streets = neatify_loop(
            new_streets,
            artifacts,
            max_segment_length=max_segment_length,
            min_dangle_length=min_dangle_length,
            clip_limit=clip_limit,
            simplification_factor=simplification_factor,
            consolidation_tolerance=consolidation_tolerance,
            eps=eps,
            angle_threshold=angle_threshold,
        )

        # This is potentially fixing some minor erroneous edges coming from Voronoi
        new_streets = induce_nodes(new_streets, eps=eps)
        new_streets = new_streets[~new_streets.geometry.normalize().duplicated()].copy()

    return new_streets


def neatify_loop(
    streets: gpd.GeoDataFrame,
    artifacts: gpd.GeoDataFrame,
    *,
    max_segment_length: float | int = 1,
    min_dangle_length: float | int = 20,
    clip_limit: float | int = 2,
    simplification_factor: float | int = 2,
    consolidation_tolerance: float | int = 10,
    angle_threshold: float = 120,
    eps: float = 1e-4,
) -> gpd.GeoDataFrame:
    """Perform an iteration of the simplification procedure which includes:
        1. Removal of false nodes
        2. Artifact classification
        3. Simplifying artifacts:
            - Single artifacts
            - Pairs of artifacts
            - Clusters of artifacts

    Parameters
    ----------
    streets : geopandas.GeoDataFrame
        Raw street network data.
    artifacts : geopandas.GeoDataFrame
        Face artifact polygons.
    max_segment_length : float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
        Used in multiple internal geometric operations.
    min_dangle_length : float | int = 20
        The threshold for determining if linestrings are dangling slivers to be
        removed or not.
    clip_limit : float | int = 2
        Following generation of the Voronoi linework, we clip to fit inside the
        polygon. To ensure we get a space to make proper topological connections
        from the linework to the actual points on the edge of the polygon, we clip
        using a polygon with a negative buffer of ``clip_limit`` or the radius of
        maximum inscribed circle, whichever is smaller.
    simplification_factor : float | int = 2
        The factor by which singles, pairs, and clusters are simplified. The
        ``max_segment_length`` is multiplied by this factor to get the
        simplification epsilon.
    consolidation_tolerance : float | int = 10
        Tolerance passed to node consolidation when generating Voronoi skeletons.
    angle_threshold : float = 120
        See the ``angle_threshold`` keyword argument in :class:`momepy.COINS`.
    eps : float = 1e-4
        Tolerance epsilon used in multiple internal geometric operations.

    Returns
    -------
    geopandas.GeoDataFrame
        The street network line data following 1 iteration of simplification.
    """

    # Remove edges fully within the artifact (dangles).
    _, r_idx = streets.sindex.query(artifacts.geometry, predicate="contains")
    # Dropping may lead to new false nodes – drop those
    streets = remove_interstitial_nodes(streets.drop(streets.index[r_idx]))

    # Filter singleton artifacts
    rook = graph.Graph.build_contiguity(artifacts, rook=True)

    # Keep only those artifacts which occur as isolates,
    # e.g. artifacts that are not part of a larger intersection
    singles = artifacts.loc[artifacts.index.intersection(rook.isolates)].copy()

    # Filter doubles
    artifacts["comp"] = rook.component_labels
    counts = artifacts["comp"].value_counts()
    doubles = artifacts.loc[artifacts["comp"].isin(counts[counts == 2].index)].copy()

    # Filter clusters
    clusters = artifacts.loc[artifacts["comp"].isin(counts[counts > 2].index)].copy()

    if not singles.empty:
        # NOTE: this drops attributes
        streets = neatify_singletons(
            singles,
            streets,
            max_segment_length=max_segment_length,
            simplification_factor=simplification_factor,
            consolidation_tolerance=consolidation_tolerance,
            angle_threshold=angle_threshold,
        )
    if not doubles.empty:
        streets = neatify_pairs(
            doubles,
            streets,
            max_segment_length=max_segment_length,
            min_dangle_length=min_dangle_length,
            clip_limit=clip_limit,
            simplification_factor=simplification_factor,
            consolidation_tolerance=consolidation_tolerance,
        )
    if not clusters.empty:
        streets = neatify_clusters(
            clusters,
            streets,
            max_segment_length=max_segment_length,
            simplification_factor=simplification_factor,
            eps=eps,
            min_dangle_length=min_dangle_length,
            consolidation_tolerance=consolidation_tolerance,
        )

    if "coins_group" in streets.columns:
        streets = streets.drop(
            columns=[c for c in streets.columns if c.startswith("coins_")]
        )
    return streets
