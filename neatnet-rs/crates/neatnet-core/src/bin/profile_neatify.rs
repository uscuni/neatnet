//! Standalone profiling binary for neatify pipeline.
//! Usage: cargo run --release --bin profile_neatify -- /path/to/edges.wkt [n_loops]

use std::fs;
use std::time::Instant;

use geo_types::LineString;
use neatnet_core::types::{EdgeStatus, NeatifyParams, StreetNetwork};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let wkt_path = args.get(1).expect("Usage: profile_neatify <edges.wkt> [n_loops]");
    let n_loops = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);

    eprintln!("Loading WKT from {}", wkt_path);
    let t0 = Instant::now();
    let content = fs::read_to_string(wkt_path).expect("Failed to read WKT file");
    let geometries: Vec<LineString<f64>> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| {
            use std::str::FromStr;
            wkt::Wkt::from_str(l)
                .ok()
                .and_then(|w| geo_types::Geometry::try_from(w).ok())
                .and_then(|g| match g {
                    geo_types::Geometry::LineString(ls) => Some(ls),
                    _ => None,
                })
        })
        .collect();
    eprintln!("Loaded {} edges in {:.3}s", geometries.len(), t0.elapsed().as_secs_f64());

    let statuses = vec![EdgeStatus::Original; geometries.len()];
    let mut network = StreetNetwork {
        geometries,
        statuses,
        attributes: None,
        crs: None,
    };

    let params = NeatifyParams {
        n_loops,
        ..NeatifyParams::default()
    };

    eprintln!("Running neatify with n_loops={}...", n_loops);
    let t0 = Instant::now();
    neatnet_core::simplify::neatify(&mut network, &params, None).unwrap();
    eprintln!("neatify completed in {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("Output: {} edges", network.geometries.len());
}
