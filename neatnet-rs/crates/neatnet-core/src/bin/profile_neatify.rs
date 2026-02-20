//! Standalone profiling binary for neatify pipeline.
//! Usage: cargo run --release --features cli --bin profile_neatify -- /path/to/edges.[wkt|parquet] [n_loops]

use std::fs;
use std::path::Path;
use std::time::Instant;

use geo_types::LineString;
use neatnet_core::types::{EdgeStatus, NeatifyParams, StreetNetwork};

fn load_wkt(path: &str) -> Vec<LineString<f64>> {
    let content = fs::read_to_string(path).expect("Failed to read WKT file");
    content
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
        .collect()
}

fn load_parquet(path: &str) -> Vec<LineString<f64>> {
    use geoarrow::array::{from_arrow_array, AsGeoArrowArray, GeoArrowArray, GeoArrowArrayAccessor};
    use geo_traits::to_geo::ToGeoLineString;

    let file = fs::File::open(path).expect("Failed to open Parquet file");
    let builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file)
            .expect("Failed to build Parquet reader");
    let reader = builder.build().expect("Failed to build batch reader");

    let mut geometries: Vec<LineString<f64>> = Vec::new();

    for batch_result in reader {
        let batch = batch_result.expect("Failed to read record batch");
        let schema = batch.schema();

        // Find the geometry column (try "geometry", then first column)
        let geom_col_idx = schema
            .column_with_name("geometry")
            .map(|(i, _)| i)
            .unwrap_or(0);

        let array_ref = batch.column(geom_col_idx);
        let field = schema.field(geom_col_idx);

        let ga = from_arrow_array(array_ref.as_ref(), field)
            .expect("Failed to parse geoarrow geometry");

        if let Some(ls_array) = ga.as_line_string_opt() {
            for i in 0..ls_array.len() {
                if !ls_array.is_null(i) {
                    if let Ok(scalar) = ls_array.value(i) {
                        geometries.push(scalar.to_line_string());
                    }
                }
            }
        }
    }

    geometries
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path = args
        .get(1)
        .expect("Usage: profile_neatify <edges.wkt|edges.parquet> [n_loops]");
    let n_loops = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);

    eprintln!("Loading from {}", input_path);
    let t0 = Instant::now();

    let geometries = if Path::new(input_path)
        .extension()
        .map_or(false, |ext| ext == "parquet")
    {
        load_parquet(input_path)
    } else {
        load_wkt(input_path)
    };

    eprintln!(
        "Loaded {} edges in {:.3}s",
        geometries.len(),
        t0.elapsed().as_secs_f64()
    );

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
