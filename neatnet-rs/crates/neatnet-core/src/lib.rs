//! neatnet-core: Pure Rust library for street network simplification.
//!
//! This crate implements the Adaptive Continuity-Preserving Simplification
//! algorithm for street networks. It detects and replaces transportation
//! artifacts (dual carriageways, roundabouts, complex intersections) with
//! morphologically meaningful centerlines.
//!
//! # Architecture
//!
//! The pipeline flows through these modules:
//!
//! 1. **nodes** – Topology fixing, node consolidation, component labeling
//! 2. **artifacts** – Face artifact detection via polygonization + KDE
//! 3. **continuity** – COINS stroke grouping
//! 4. **geometry** – Voronoi skeleton generation
//! 5. **simplify** – Main pipeline orchestration
//! 6. **gaps** – Gap closing and line extension
//! 7. **spatial** – R*-tree spatial index utilities
//!
//! # Usage
//!
//! ```ignore
//! use neatnet_core::types::{StreetNetwork, NeatifyParams};
//! use neatnet_core::simplify::neatify;
//!
//! let mut network = StreetNetwork { /* ... */ };
//! let params = NeatifyParams::default();
//! neatify(&mut network, &params, None).unwrap();
//! ```

pub mod artifacts;
pub mod continuity;
pub mod gaps;
pub mod geometry;
pub mod nodes;
pub mod ops;
pub mod simplify;
pub mod spatial;
pub mod types;

// Re-export key types at crate root for convenience
pub use simplify::neatify;
pub use types::{EdgeStatus, NeatifyParams, StreetNetwork};
