"""Utility functions"""

import geopandas
import osmnx


def fetch_milton_keynes() -> geopandas.GeoDataFrame:
    """Query for Milton Keynes network drive edges -- used in examples"""

    osm_graph = osmnx.graph_from_place("Milton Keynes", network_type="drive")
    osm_graph = osmnx.projection.project_graph(osm_graph, to_crs=27700)

    return osmnx.graph_to_gdfs(
        osmnx.convert.to_undirected(osm_graph),
        nodes=False,
        edges=True,
        node_geometry=False,
        fill_edge_geometry=True,
    ).reset_index(drop=True)
