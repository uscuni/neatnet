import math
import operator

import geopandas as gpd
import numpy as np
import shapely

__all__ = [
    "close_gaps",
    "extend_lines",
]


def close_gaps(
    gdf: gpd.GeoSeries | gpd.GeoDataFrame, tolerance: float
) -> gpd.GeoSeries:
    """Close gaps in LineString geometry where it should be contiguous.
    Snaps both lines to a centroid of a gap in between.

    Parameters
    ----------
    gdf : geopandas.GeoSeries | geopandas.GeoDataFrame
        LineString representations of a network.
    tolerance : float
        Nodes within ``tolerance`` will be snapped together.

    Returns
    -------
    geopandas.GeoSeries

    See also
    --------
    neatnet.extend_lines
    neatnet.remove_interstitial_nodes
    """

    geom = gdf.geometry.array
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = shapely.points(np.unique(coords[edges], axis=0))

    buffered = shapely.buffer(points, tolerance / 2)

    dissolved = shapely.union_all(buffered)

    exploded = [
        shapely.get_geometry(dissolved, i)
        for i in range(shapely.get_num_geometries(dissolved))
    ]

    centroids = shapely.centroid(exploded)

    snapped = shapely.snap(geom, shapely.union_all(centroids), tolerance)

    return gpd.GeoSeries(snapped, crs=gdf.crs)


def extend_lines(
    gdf: gpd.GeoDataFrame,
    tolerance: float,
    *,
    target: None | gpd.GeoSeries | gpd.GeoDataFrame = None,
    barrier: None | gpd.GeoSeries | gpd.GeoDataFrame = None,
    extension: int | float = 0,
) -> gpd.GeoDataFrame:
    """Extends lines from ``gdf`` to itself or target within a set tolerance.

    Extends unjoined ends of LineString segments to join with other segments or target.
    If ``target`` is passed, extend lines to target. Otherwise extend lines to itself.

    If ``barrier`` is passed, each extended line is checked for intersection with
    ``barrier``. If they intersect, extended line is not returned. This can be
    useful if you don't want to extend street network segments through buildings.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing LineString geometry.
    tolerance : float
        Tolerance in snapping (by how much could be each segment extended).
    target : None | geopandas.GeoSeries | geopandas.GeoDataFrame
        Target geometry to which ``gdf`` gets extended.
        Has to be (Multi)LineString geometry.
    barrier : None | geopandas.GeoSeries | geopandas.GeoDataFrame = None
        Extended line is not used if it intersects barrier.
    extension : int | float = 0
        By how much to extend line beyond the snapped geometry. Useful
        when creating enclosures to avoid floating point imprecision.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with extended geometries.

    See also
    --------
    neatnet.close_gaps
    neatnet.remove_interstitial_nodes
    """

    # explode to avoid MultiLineStrings
    # reset index due to the bug in GeoPandas explode
    df = gdf.reset_index(drop=True).explode(ignore_index=True)

    if target is None:
        target = df
        itself = True
    else:
        itself = False

    # get underlying shapely geometry
    geom = df.geometry.array

    # extract array of coordinates and number per geometry
    coords = shapely.get_coordinates(geom)
    indices = shapely.get_num_coordinates(geom)

    # generate a list of start and end coordinates and create point geometries
    edges = [0]
    i = 0
    for ind in indices:
        ix = i + ind
        edges.append(ix - 1)
        edges.append(ix)
        i = ix
    edges = edges[:-1]
    points = shapely.points(np.unique(coords[edges], axis=0))

    # query LineString geometry to identify points intersecting 2 geometries
    tree = shapely.STRtree(geom)
    inp, res = tree.query(points, predicate="intersects")
    unique, counts = np.unique(inp, return_counts=True)
    ends = np.unique(res[np.isin(inp, unique[counts == 1])])

    new_geoms = []
    # iterate over cul-de-sac-like segments and attempt to snap them to street network
    for line in ends:
        l_coords = shapely.get_coordinates(geom[line])

        start = shapely.points(l_coords[0])
        end = shapely.points(l_coords[-1])

        first = list(tree.query(start, predicate="intersects"))
        second = list(tree.query(end, predicate="intersects"))
        first.remove(line)
        second.remove(line)

        t = target if not itself else target.drop(line)

        if first and not second:
            snapped = _extend_line(l_coords, t, tolerance)
            if (
                barrier is not None
                and barrier.sindex.query(
                    shapely.linestrings(snapped), predicate="intersects"
                ).size
                > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(shapely.linestrings(snapped))
                else:
                    new_geoms.append(
                        shapely.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )
        elif not first and second:
            snapped = _extend_line(np.flip(l_coords, axis=0), t, tolerance)
            if (
                barrier is not None
                and barrier.sindex.query(
                    shapely.linestrings(snapped), predicate="intersects"
                ).size
                > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(shapely.linestrings(snapped))
                else:
                    new_geoms.append(
                        shapely.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )
        elif not first and not second:
            one_side = _extend_line(l_coords, t, tolerance)
            one_side_e = _extend_line(one_side, t, extension, snap=False)
            snapped = _extend_line(np.flip(one_side_e, axis=0), t, tolerance)
            if (
                barrier is not None
                and barrier.sindex.query(
                    shapely.linestrings(snapped), predicate="intersects"
                ).size
                > 0
            ):
                new_geoms.append(geom[line])
            else:
                if extension == 0:
                    new_geoms.append(shapely.linestrings(snapped))
                else:
                    new_geoms.append(
                        shapely.linestrings(
                            _extend_line(snapped, t, extension, snap=False)
                        )
                    )

    df.iloc[ends, df.columns.get_loc(df.geometry.name)] = new_geoms
    return df


def _extend_line(
    coords: np.ndarray,
    target: gpd.GeoDataFrame | gpd.GeoSeries,
    tolerance: float,
    snap: bool = True,
) -> np.ndarray:
    """Extends a line geometry to snap on the target within a tolerance."""

    if snap:
        extrapolation = _get_extrapolated_line(
            coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(),
            tolerance,
        )
        int_idx = target.sindex.query(extrapolation, predicate="intersects")
        intersection = shapely.intersection(
            target.iloc[int_idx].geometry.array, extrapolation
        )
        if intersection.size > 0:
            if len(intersection) > 1:
                distances = {}
                ix = 0
                for p in intersection:
                    distance = shapely.distance(p, shapely.points(coords[-1]))
                    distances[ix] = distance
                    ix = ix + 1
                minimal = min(distances.items(), key=operator.itemgetter(1))[0]
                new_point_coords = shapely.get_coordinates(intersection[minimal])

            else:
                new_point_coords = shapely.get_coordinates(intersection[0])
            coo = np.append(coords, new_point_coords)
            new = np.reshape(coo, (len(coo) // 2, 2))

            return new
        return coords

    extrapolation = _get_extrapolated_line(
        coords[-4:] if len(coords.shape) == 1 else coords[-2:].flatten(),
        tolerance,
        point=True,
    )
    return np.vstack([coords, extrapolation])


def _get_extrapolated_line(
    coords: np.ndarray, tolerance: float, point: bool = False
) -> tuple[float, float] | shapely.LineString:
    """Creates a shapely line extrapolated in p1->p2 direction."""

    p1 = coords[:2]
    p2 = coords[2:]
    a = p2

    # defining new point based on the vector between existing points
    if p1[0] >= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] >= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            - tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    elif p1[0] <= p2[0] and p1[1] <= p2[1]:
        b = (
            p2[0]
            + tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    else:
        b = (
            p2[0]
            - tolerance
            * math.cos(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
            p2[1]
            + tolerance
            * math.sin(
                math.atan(
                    math.fabs(p1[1] - p2[1] + 0.000001)
                    / math.fabs(p1[0] - p2[0] + 0.000001)
                )
            ),
        )
    if point:
        return b
    return shapely.linestrings([a, b])
