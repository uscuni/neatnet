//! Spatial index utilities built on `rstar` R*-tree.

use geo::BoundingRect;
use geo_types::{Coord, LineString, Polygon};
use rstar::{RTree, RTreeObject, AABB};

/// A line segment stored in the R*-tree, carrying an index back to the
/// original geometry collection.
#[derive(Debug, Clone)]
pub struct IndexedEnvelope {
    /// Index of the geometry in the source collection.
    pub index: usize,
    /// Axis-aligned bounding box (min_x, min_y, max_x, max_y).
    pub min: [f64; 2],
    pub max: [f64; 2],
}

impl RTreeObject for IndexedEnvelope {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners(self.min, self.max)
    }
}

/// Build an R*-tree from LineString geometries using their bounding rects.
pub fn build_rtree(geometries: &[LineString<f64>]) -> RTree<IndexedEnvelope> {
    let items: Vec<IndexedEnvelope> = geometries
        .iter()
        .enumerate()
        .filter_map(|(i, geom)| {
            let rect = geom.bounding_rect()?;
            Some(IndexedEnvelope {
                index: i,
                min: [rect.min().x, rect.min().y],
                max: [rect.max().x, rect.max().y],
            })
        })
        .collect();

    RTree::bulk_load(items)
}

/// Build an R*-tree from Polygon geometries using their bounding rects.
pub fn build_rtree_polys(geometries: &[Polygon<f64>]) -> RTree<IndexedEnvelope> {
    let items: Vec<IndexedEnvelope> = geometries
        .iter()
        .enumerate()
        .filter_map(|(i, geom)| {
            let rect = geom.bounding_rect()?;
            Some(IndexedEnvelope {
                index: i,
                min: [rect.min().x, rect.min().y],
                max: [rect.max().x, rect.max().y],
            })
        })
        .collect();

    RTree::bulk_load(items)
}

/// Query the R*-tree for geometries whose envelope intersects the given point.
pub fn query_point(tree: &RTree<IndexedEnvelope>, point: Coord<f64>) -> Vec<usize> {
    let p = [point.x, point.y];
    tree.locate_in_envelope_intersecting(&AABB::from_point(p))
        .map(|item| item.index)
        .collect()
}

/// Query the R*-tree for geometries whose envelope intersects the given bounding box.
pub fn query_envelope(
    tree: &RTree<IndexedEnvelope>,
    min: [f64; 2],
    max: [f64; 2],
) -> Vec<usize> {
    tree.locate_in_envelope_intersecting(&AABB::from_corners(min, max))
        .map(|item| item.index)
        .collect()
}

/// Query the R*-tree for geometries within `distance` of a point (envelope-based).
pub fn query_dwithin(
    tree: &RTree<IndexedEnvelope>,
    point: Coord<f64>,
    distance: f64,
) -> Vec<usize> {
    let min = [point.x - distance, point.y - distance];
    let max = [point.x + distance, point.y + distance];
    query_envelope(tree, min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_envelope() {
        let item = IndexedEnvelope {
            index: 42,
            min: [0.0, 0.0],
            max: [1.0, 1.0],
        };
        let env = item.envelope();
        assert_eq!(env.lower(), [0.0, 0.0]);
        assert_eq!(env.upper(), [1.0, 1.0]);
    }

    #[test]
    fn test_query_point_basic() {
        let items = vec![
            IndexedEnvelope {
                index: 0,
                min: [0.0, 0.0],
                max: [2.0, 2.0],
            },
            IndexedEnvelope {
                index: 1,
                min: [3.0, 3.0],
                max: [5.0, 5.0],
            },
        ];
        let tree = RTree::bulk_load(items);
        let results = query_point(&tree, Coord { x: 1.0, y: 1.0 });
        assert_eq!(results, vec![0]);

        let results = query_point(&tree, Coord { x: 4.0, y: 4.0 });
        assert_eq!(results, vec![1]);

        let results = query_point(&tree, Coord { x: 10.0, y: 10.0 });
        assert!(results.is_empty());
    }
}
