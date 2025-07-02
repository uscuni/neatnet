# FUA testing data

## Contents

* This `README.md`
* `generate_simplified.py` – see [neatnet#7](https://github.com/uscuni/neatnet/issues/7)
* Data
   * There is a directory for each FUA listed below that contains 2 files:
      * `original.parquet`: The original input street network derived from [OSM](https://www.openstreetmap.org/about) via [OSMNX](https://osmnx.readthedocs.io/en/stable/).
      * `simplified.parquet`: The simplified street network following our algorithm with *default parameters*.
* Known vs. Observed simplifed comparison notebook
   1. Download CI artifact zip
   2. Put in the same directory
   3. Run

## FUA Information

| FUA  | City                                   | Shortand              |
| ---  | ---                                    | ---                   |
| 1133 | Aleppo, Syria, Middle East / Asia      | `aleppo_1133`         |
| 869  | Auckland, New Zealand, Oceania / Asia  | `auckland_869`        |
| 809  | Douala, Cameroon, Africa               | `douala_809`          |
| 1656 | Liège, Belgium, Europe                 | `liege_1656`          |
| 4617 | Bucaramanga, Colombia, S. America      | `bucaramanga_4617`    |
| 4881 | Salt Lake City, Utah, USA, N. America  | `slc_4881`            |

---------------------------------------

Copyright (c) 2024-, neatnet Developers
