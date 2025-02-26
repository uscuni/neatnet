"""Geometry-related functions"""

import collections
import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from libpysal import graph
from scipy import spatial

from .nodes import consolidate_nodes


def _is_within(
    line: np.ndarray, poly: shapely.Polygon, rtol: float = 1e-4
) -> np.ndarray:
    """Check if the line is within a polygon with a set relative tolerance.

    Parameters
    ----------
    line : np.ndarray[shapely.LineString]
        Input line to check relationship.
    poly : shapely.Polygon
        Input polygon to check relationship.
    rtol : float = -1e4
        The set relative tolerance.

    Returns
    -------
    np.ndarray
        ``True`` if ``line`` is either entirely within ``poly`` or if
        ``line`` is within `poly`` based on a relaxed ``rtol`` relative tolerance.
    """

    within = shapely.within(line, poly)
    if within.all():
        return within

    intersection = shapely.intersection(line, poly)
    return np.abs(shapely.length(intersection) - shapely.length(line)) <= rtol


def angle_between_two_lines(
    line1: shapely.LineString, line2: shapely.LineString
) -> float:
    """Return the angle between 2 two lines (assuming they share a vertex).
    Based on ``momepy.coins`` but adapted to shapely lines.
    """

    return_bad = 0.0

    lines_distinct = line1 != line2
    if not lines_distinct:
        warnings.warn(
            f"Input lines are identical - must be distinct. Returning {return_bad}.",
            UserWarning,
            stacklevel=2,
        )
        return return_bad

    # extract points
    a, b, c, d = shapely.get_coordinates([line1, line2]).tolist()
    a, b, c, d = tuple(a), tuple(b), tuple(c), tuple(d)

    # assertion: we expect exactly 2 of the 4 points to be identical
    # (lines touch in this point)
    points = collections.Counter([a, b, c, d])

    lines_share_vertex = max(points.values()) > 1
    if not lines_share_vertex:
        warnings.warn(
            f"Input lines do not share a vertex. Returning {return_bad}.",
            UserWarning,
            stacklevel=2,
        )
        return return_bad

    # points where line touch = "origin" (for vector-based angle calculation)
    origin = [k for k, v in points.items() if v == 2][0]
    # other 2 unique points (one on each line)
    point1, point2 = (k for k, v in points.items() if v == 1)

    # translate lines into vectors (numpy arrays)
    v1 = [point1[0] - origin[0], point1[1] - origin[1]]
    v2 = [point2[0] - origin[0], point2[1] - origin[1]]

    # compute angle between 2 vectors in degrees
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    cos_theta = round(dot_product / (norm_v1 * norm_v2), 6)  # precision issues fix
    angle = math.degrees(math.acos(cos_theta))

    return angle


def voronoi_skeleton(
    lines: list | np.ndarray | gpd.GeoSeries,
    poly: None | shapely.Polygon = None,
    snap_to: None | gpd.GeoSeries = None,
    max_segment_length: float | int = 1,
    buffer: None | float | int = None,
    secondary_snap_to: None | gpd.GeoSeries = None,
    clip_limit: None | float | int = 2,
    consolidation_tolerance: None | float | int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns average geometry.

    Parameters
    ----------
    lines : list | numpy.ndarray | geopandas.GeoSeries
        LineStrings connected at endpoints. If ``poly`` is passed in, ``lines``
        must be a ``geopandas.GeoSeries``.
    poly : None | shapely.Polygon = None
        Polygon enclosed by ``lines``.
    snap_to : None | gpd.GeoSeries = None
        Series of geometries that shall be connected to the skeleton.
    max_segment_length: float | int = 1
        Additional vertices will be added so that all line segments
        are no longer than this value. Must be greater than 0.
    buffer : None | float | int = None
        Optional custom buffer distance for dealing with Voronoi infinity issues.
    secondary_snap_to : None | gpd.GeoSeries = None
        Fall-back series of geometries that shall be connected to the skeleton.
    clip_limit : None | float | int = 2
        Following generation of the Voronoi linework, we clip to fit inside the polygon.
        To ensure we get a space to make proper topological connections from the
        linework to the actual points on the edge of the polygon, we clip using a
        polygon with a negative buffer of ``clip_limit`` or the radius of
        maximum inscribed circle, whichever is smaller.
    consolidation_tolerance : None | float | int = None
        Tolerance passed to node consolidation within the resulting skeleton.
        If ``None``, no consolidation happens.

    Returns
    -------
    edgelines : numpy.ndarray
        Array of averaged geometries.
    splitters : numpy.ndarray
        Split points.
    """
    if buffer is None:
        buffer = max_segment_length * 20
    if not poly:
        if not isinstance(lines, gpd.GeoSeries):
            lines = gpd.GeoSeries(lines)
        poly = shapely.box(*lines.total_bounds)
    # get an additional line around the lines to avoid infinity issues with Voronoi
    extended_lines = list(lines) + [poly.buffer(buffer).boundary]

    # interpolate lines to represent them as points for Voronoi
    shapely_lines = extended_lines
    points, ids = shapely.get_coordinates(
        shapely.segmentize(shapely_lines, max_segment_length), return_index=True
    )

    # remove duplicated coordinates
    unq, count = np.unique(points, axis=0, return_counts=True)
    mask = np.isin(points, unq[count > 1]).all(axis=1)
    points = points[~mask]
    ids = ids[~mask]

    # generate Voronoi diagram
    voronoi_diagram = spatial.Voronoi(points)

    # get all rigdes and filter only those between the two lines
    pts = voronoi_diagram.ridge_points
    mapped = np.take(ids, pts)
    rigde_vertices = np.array(voronoi_diagram.ridge_vertices)

    # iterate over segment-pairs and keep rigdes between input geometries
    _edgelines = []
    to_add = []
    splitters = []

    # determine the negative buffer distance to avoid overclipping of narrow polygons
    # this can still result in some missing links, but only in rare cases
    dist = min([clip_limit, shapely.ops.polylabel(poly).distance(poly.boundary) * 0.4])
    limit = poly.buffer(-dist)

    # drop ridges that are between points coming from the same line
    selfs = mapped[:, 0] == mapped[:, 1]
    buff = (mapped == mapped.max()).any(axis=1)
    mapped = mapped[~(selfs | buff)]
    rigde_vertices = rigde_vertices[~(selfs | buff)]
    unique = np.unique(np.sort(mapped, axis=1), axis=0)

    for a, b in unique:
        mask = ((mapped[:, 0] == a) | (mapped[:, 0] == b)) & (
            (mapped[:, 1] == a) | (mapped[:, 1] == b)
        )

        verts = rigde_vertices[mask]

        # generate the line in between the lines
        edgeline = shapely.line_merge(
            shapely.multilinestrings(voronoi_diagram.vertices[verts])
        )

        # check if the edgeline is within polygon
        if not edgeline.within(limit):
            if not isinstance(edgeline, shapely.MultiLineString):
                # if not, clip it by the polygon with a small negative buffer to keep
                # the gap between edgeline and poly boundary to avoid possible
                # overlapping lines
                edgeline = shapely.intersection(edgeline, limit)

                # in edge cases, this can result in a MultiLineString with one sliver
                # part
                edgeline = _remove_sliver(edgeline)
            # if the edgeline is a MultiLineString, treat each part independently
            else:
                parts = []
                for part in edgeline.geoms:
                    part = shapely.intersection(part, limit)
                    part = _remove_sliver(part)
                    if not part.is_empty:
                        parts.append(part)
                edgeline = shapely.MultiLineString(parts)

        # check if a, b lines share a node
        intersection = shapely_lines[b].intersection(shapely_lines[a])
        # if they do, add shortest line from the edgeline to the shared node and
        # combine it with the edgeline. Also, avoid an inner loop in more complex input
        # that would create connection across
        if not intersection.is_empty and not (
            intersection.geom_type == "MultiPoint"
            and (len(intersection.geoms) == 2 and len(lines) != 2)
        ):
            # we need union of edgeline and shortest because snap is buggy in GEOS
            # and line_merge as well. This results in a MultiLineString but we can
            # deal with those later. For now, we just need this extended edgeline to
            # be a single geometry to ensure the component discovery below works as
            # intended
            # get_parts is needed as in case of voronoi based on two lines, these
            # intersect on both ends, hence both need to be extended
            edgeline = shapely.union(
                edgeline,
                shapely.union_all(
                    shapely.shortest_line(
                        shapely.get_parts(intersection), edgeline.boundary
                    )
                ),
            )
        # add final edgeline to the list
        _edgelines.append(edgeline)

    edgelines = np.array(_edgelines)[~(shapely.is_empty(_edgelines))]

    if edgelines.shape[0] > 0:
        # if there is no explicit snapping target, snap to the boundary of the polygon
        # via the shortest line. That is by definition always within the polygon
        # (Martin thinks) (James concurs)
        if snap_to is not False:
            if snap_to is None:
                sl = shapely.shortest_line(
                    shapely.union_all(edgelines).boundary, poly.boundary
                )
                to_add.append(sl)
                splitters.append(shapely.get_point(sl, -1))

            # if we have some snapping targets, we need to figure out
            # what shall be snapped to what
            else:
                additions, splits = snap_to_targets(
                    edgelines, poly, snap_to, secondary_snap_to
                )
                to_add.extend(additions)
                splitters.extend(splits)

            # concatenate edgelines and their additions snapping to edge
            edgelines = np.concatenate([edgelines, to_add])
        # simplify to avoid unnecessary point density and some wobbliness
        edgelines = shapely.simplify(edgelines, max_segment_length)
    # drop empty
    edgelines = edgelines[edgelines != None]  # noqa: E711

    edgelines = shapely.line_merge(edgelines[shapely.length(edgelines) > 0])
    edgelines = _as_parts(edgelines)
    edgelines = _consolidate(edgelines, consolidation_tolerance)

    return edgelines, np.array(splitters)


def _remove_sliver(
    edgeline: shapely.LineString | shapely.MultiLineString,
) -> shapely.LineString:
    """Remove sliver(s) if present."""
    if edgeline.geom_type == "MultiLineString":
        parts = shapely.get_parts(edgeline)
        edgeline = parts[np.argmax(shapely.length(parts))]
    return edgeline


def _as_parts(edgelines: np.ndarray) -> np.ndarray:
    """Return constituent LineStrings if MultiLineString present."""
    if np.unique(shapely.get_type_id(edgelines)).shape[0] > 1:
        edgelines = shapely.get_parts(edgelines)
    return edgelines


def _consolidate(
    edgelines: np.ndarray, consolidation_tolerance: None | float | int
) -> np.ndarray:
    """Return ``edgelines`` from consolidated nodes, if criteria met."""
    if consolidation_tolerance and edgelines.shape[0] > 0:
        edgelines = consolidate_nodes(
            edgelines, tolerance=consolidation_tolerance, preserve_ends=True
        ).geometry.to_numpy()
    return edgelines


def snap_to_targets(
    edgelines: np.ndarray,
    poly: shapely.Polygon,
    snap_to: gpd.GeoSeries,
    secondary_snap_to: None | gpd.GeoSeries = None,
) -> tuple[list[shapely.LineString], list[shapely.Point]]:
    """Snap edgelines to vertices.

    Parameters
    ----------
    edgelines : numpy.ndarray
        Voronoi skeleton edges.
    poly : None | shapely.Polygon = None
        Polygon enclosed by ``lines``.
    snap_to : None | gpd.GeoSeries = None
        Series of geometries that shall be connected to the skeleton.
    secondary_snap_to : None | gpd.GeoSeries = None
        Fall-back series of geometries that shall be connected to the skeleton.

    Returns
    -------
    to_add, to_split : tuple[list[shapely.LineString], list[shapely.Point]]
        Lines to add and points where to split.
    """

    to_add: list = []
    to_split: list = []

    # generate graph from lines
    comp_labels, comp_counts, components = _prep_components(edgelines)

    primary_union = shapely.union_all(snap_to)
    secondary_union = shapely.union_all(secondary_snap_to)

    # if there are muliple components, loop over all and treat each
    if len(components) > 1:
        for comp_label, comp in components.geometry.items():
            cbound = comp.boundary

            # if component does not interest the boundary, it needs to be snapped
            # if it does but has only one part, this part interesect only on one
            # side (the node remaining from the removed edge) and needs to be
            # snapped on the other side as well
            if (
                (not comp.intersects(poly.boundary))
                or comp_counts[comp_label] == 1
                or (
                    not comp.intersects(primary_union)
                )  # ! this fixes one thing but may break others
            ):
                # add segment composed of the shortest line to the nearest snapping
                # target. We use boundary to snap to endpoints of edgelines only
                sl = shapely.shortest_line(cbound, primary_union)
                if _is_within(sl, poly):
                    to_split, to_add = _split_add(sl, to_split, to_add)
                else:
                    if secondary_snap_to is not None:
                        sl = shapely.shortest_line(cbound, secondary_union)
                        to_split, to_add = _split_add(sl, to_split, to_add)
    else:
        # if there is a single component, ensure it gets a shortest line to an
        # endpoint from each snapping target
        for target in snap_to:
            sl = shapely.shortest_line(components.boundary.item(), target)
            if _is_within(sl, poly):
                to_split, to_add = _split_add(sl, to_split, to_add)
            else:
                warnings.warn(
                    "Could not create a connection as it would lead outside "
                    "of the artifact.",
                    UserWarning,
                    stacklevel=2,
                )
    return to_add, to_split


def _prep_components(
    lines: np.ndarray | gpd.GeoSeries,
) -> tuple[pd.Series, pd.Series, gpd.GeoSeries]:
    """Helper for preparing graph components & labels in PySAL."""

    # cast edgelines to gdf
    lines = gpd.GeoDataFrame(geometry=lines)

    # build queen contiguity on edgelines and extract component labels
    not_empty = ~lines.is_empty
    not_nan = ~lines.geometry.isna()
    lines = lines[not_empty | not_nan]
    comp_labels = graph.Graph.build_contiguity(lines, rook=False).component_labels

    # compute size of each component
    comp_counts = comp_labels.value_counts()

    # get MultiLineString geometry per connected component
    components = lines.dissolve(comp_labels)

    return comp_labels, comp_counts, components


def _split_add(line: shapely.LineString, splits: list, adds: list) -> tuple[list, list]:
    """Helper for preparing splitter points & added lines."""
    splits.append(shapely.get_point(line, -1))
    adds.append(line)
    return splits, adds
