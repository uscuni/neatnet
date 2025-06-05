import geopandas
import momepy


def continuity(
    streets: geopandas.GeoDataFrame, angle_threshold: float = 120
) -> tuple[geopandas.GeoDataFrame, momepy.COINS]:
    """Assign COINS-based information to streets.

    Parameters
    ----------
    streets : geopandas.GeoDataFrame
        Street network.
    angle_threshold : float = 120
        See the ``angle_threshold`` keyword argument in ``momepy.COINS()``.

    Returns
    -------
    streets : geopandas.GeoDataFrame
        The input ``streets`` with additional columns describing COINS information.
    coins : momepy.COINS
        **This is not used in production.**

    Notes
    -----
    The returned ``coins`` object is not used in production, but is
    very helpful in testing & debugging. See gh:neatnet#49.
    """
    streets = streets.copy()

    # Measure continuity of street network
    coins = momepy.COINS(streets, angle_threshold=angle_threshold, flow_mode=True)

    # Assing continuity group
    group, end = coins.stroke_attribute(True)
    streets["coins_group"] = group
    streets["coins_end"] = end

    # Assign length of each continuity group and a number of segments within the group.
    coins_grouped = streets.length.groupby(streets.coins_group)
    streets["coins_len"] = coins_grouped.sum()[streets.coins_group].values
    streets["coins_count"] = coins_grouped.size()[streets.coins_group].values

    return streets, coins


def get_stroke_info(
    artifacts: geopandas.GeoSeries | geopandas.GeoDataFrame,
    streets: geopandas.GeoSeries | geopandas.GeoDataFrame,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Generate information about strokes within ``artifacts`` and the
    resulting lists can be assigned as columns to ``artifacts``. Classifies
    the strokes within the CES typology.

        * 'continuing' strokes - continues before and after artifact.
        * 'ending' strokes - continues only at one end.
        * 'single' strokes - does not continue.

    Parameters
    ----------
    artifacts : geopandas.GeoSeries | geopandas.GeoDataFrame
        Polygons representing the artifacts.
    streets : geopandas.GeoSeries | geopandas.GeoDataFrame
        LineStrings representing the street network.

    Returns
    -------
    strokes : list[int]
        All strokes counts.
    c_ : list[int]
        Counts for 'continuing' strokes - continues before and after artifact.
    e_ : list[int]
        Counts for 'ending' strokes - continues only at one end.
    s_ : list[int]
        Counts for 'single' strokes - does not continue.
    """
    strokes = []
    c_ = []
    e_ = []
    s_ = []
    for geom in artifacts.geometry:
        singles = 0
        ends = 0
        edges = streets.iloc[streets.sindex.query(geom, predicate="covers")]
        ecg = edges.coins_group
        if ecg.nunique() == 1 and edges.shape[0] == edges.coins_count.iloc[0]:
            # roundabout special case
            singles = 1
            mains = 0
        else:
            all_ends = edges[edges.coins_end]
            ae_cg = all_ends.coins_group
            mains = edges[~ecg.isin(ae_cg)].coins_group.nunique()
            visited = []
            for coins_count, group in zip(all_ends.coins_count, ae_cg, strict=True):
                if group not in visited:
                    if coins_count == (ecg == group).sum():
                        singles += 1
                        visited.append(group)
                    else:
                        # do not add to visited -- may be disjoint within the artifact
                        ends += 1
        strokes.append(ecg.nunique())
        c_.append(mains)
        e_.append(ends)
        s_.append(singles)
    return strokes, c_, e_, s_
