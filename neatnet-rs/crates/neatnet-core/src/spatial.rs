//! Spatial index utilities built on `rstar` R*-tree.

use geo_types::Coord;
use geos::Geom;
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

/// Build an R*-tree from GEOS geometries using their envelopes.
///
/// Each geometry is represented by its bounding box with an index
/// back to its position in the input slice.
pub fn build_rtree(geometries: &[geos::Geometry]) -> RTree<IndexedEnvelope> {
    let items: Vec<IndexedEnvelope> = geometries
        .iter()
        .enumerate()
        .filter_map(|(i, geom)| {
            // Get the envelope (bounding box) from GEOS
            let envelope = geom.envelope().ok()?;
            let env_coords = envelope.get_coord_seq().ok()?;
            if env_coords.size().ok()? == 0 {
                return None;
            }

            // For a point, envelope is a single point; for lines/polys it's a polygon
            // Use the geometry's own min/max coordinates
            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;

            let n = env_coords.size().ok()?;
            for j in 0..n {
                let x = env_coords.get_x(j).ok()?;
                let y = env_coords.get_y(j).ok()?;
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }

            Some(IndexedEnvelope {
                index: i,
                min: [min_x, min_y],
                max: [max_x, max_y],
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
