import geopandas
import momepy


def continuity(
    roads: geopandas.GeoDataFrame, angle_threshold: float = 120
) -> tuple[geopandas.GeoDataFrame, momepy.COINS]:
    """Assign COINS-based information to streets.

    Parameters
    ----------
    roads :  geopandas.GeoDataFrame
        Street network.
    angle_threshold : float = 120
        See the ``angle_threshold`` keyword argument in ``momepy.COINS()``.

    Returns
    -------
    roads : geopandas.GeoDataFrame
        The input ``roads`` with additional columns describing COINS information.
    coins : momepy.COINS
        **This is not used in production.**

    Notes
    -----
    The returned ``coins`` object is not used in production, but is
    very helpful in testing & debugging. See gh:neatnet#49.
    """
    roads = roads.copy()

    # Measure continuity of street network
    coins = momepy.COINS(roads, angle_threshold=angle_threshold, flow_mode=True)

    # Assing continuity group
    group, end = coins.stroke_attribute(True)
    roads["coins_group"] = group
    roads["coins_end"] = end

    # Assign length of each continuity group and a number of segments within the group.
    coins_grouped = roads.length.groupby(roads.coins_group)
    roads["coins_len"] = coins_grouped.sum()[roads.coins_group].values
    roads["coins_count"] = coins_grouped.size()[roads.coins_group].values

    return roads, coins


def get_stroke_info(
    artifacts: geopandas.GeoSeries | geopandas.GeoDataFrame,
    roads: geopandas.GeoSeries | geopandas.GeoDataFrame,
    typify: bool = False,
) -> tuple[list[int], list[int], list[int], list[int]] | geopandas.GeoDataFrame:
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
    roads : geopandas.GeoSeries | geopandas.GeoDataFrame
        LineStrings representing the street network.
    typify : bool = False
        ....

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
    roads : geopandas.GeoDataFrame
        ...
    """
    strokes = []
    c_ = []
    e_ = []
    s_ = []

    if typify:
        roads["typology"] = "x"

    for geom in artifacts.geometry:
        singles = 0
        ends = 0
        edges = roads.iloc[roads.sindex.query(geom, predicate="covers")]
        ecg = edges.coins_group
        mains_index = []
        if typify:
            ends_index = []
            singles_index = []
        if ecg.nunique() == 1 and edges.shape[0] == edges.coins_count.iloc[0]:
            # roundabout special case
            singles = 1
            mains = 0
        else:
            all_ends = edges[edges.coins_end]
            ae_cg = all_ends.coins_group

            mains_slice = edges[~ecg.isin(ae_cg)]
            mains_index += mains_slice.index.to_list()
            mains = mains_slice.coins_group.nunique()

            visited = []
            for ix, coins_count, group in zip(
                all_ends.index, all_ends.coins_count, ae_cg, strict=True
            ):
                if group not in visited:
                    if coins_count == (ecg == group).sum():
                        singles += 1
                        visited.append(group)
                        if typify:
                            singles_index += [ix]
                    else:
                        # do not add to visited -- may be disjoint within the artifact
                        ends += 1
                        if typify:
                            ends_index += [ix]
        strokes.append(ecg.nunique())
        c_.append(mains)
        e_.append(ends)
        s_.append(singles)
        if typify:
            roads.loc[mains_index, "typology"] = "C"
            roads.loc[ends_index, "typology"] = "E"
            roads.loc[singles_index, "typology"] = "S"

    if typify:
        group_mapper = dict(
            roads[roads["typology"] != "x"][["coins_group", "typology"]].values
        )
        roads["typology"] = roads["coins_group"].map(group_mapper)
        return roads
    else:
        return strokes, c_, e_, s_
