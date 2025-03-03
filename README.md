# `neatnet`: Street Geometry Processing Toolkit

[![Continuous Integration](https://github.com/uscuni/neatnet/actions/workflows/testing.yml/badge.svg)](https://github.com/uscuni/neatnet/actions/workflows/testing.yml) [![codecov](https://codecov.io/gh/uscuni/neatnet/graph/badge.svg?token=GFISMU0WPS)](https://codecov.io/gh/uscuni/neatnet)

## Introduction

`neatnet` offers a set of tools pre-processing of street network geometry aimed at its simplification. This typically means removal of dual carrieageways, roundabouts and similar transportation-focused geometries and their replacement with a new geometry representing the street space via its centerline. The resulting geometry shall be closer to a morphological representation of space than the original source, that is typically drawn with transportation in mind (e.g. OpenStreetMap).

## Examples

```py
import neatnet

simplified = neatnet.simplify_network(gdf)
```

## Contribution

While we consider the API stable, the project is young and may be evolving fast. All contributions are very welcome, see our guidelines in [`CONTRIBUTING.md`](https://github.com/uscuni/neatnet/blob/main/CONTRIBUTING.md).

## Recommended Citations

The package is a result of a scientific collaboration between [The Research Team on Urban Structure](https://uscuni.org) of Charles University (USCUNI), [NEtwoRks, Data, and Society](https://nerds.itu.dk) research group of IT University Copenhagen (NERDS) and [Oak Ridge National Laboratory](https://www.ornl.gov/gshsd).

If you use `neatnet` for a research purpose, please consider citing the original paper introducing it.

### Canonical Citation (primary)

*forthcoming*

### Repository Citation (secondary)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14765801.svg)](https://doi.org/10.5281/zenodo.14765801)

* **Fleischmann, M., Vybornova, A., & Gaboardi, J.D.** (2025). `uscuni/neatnet`. Zenodo. https://doi.org/10.5281/zenodo.14765801

---------------------------------------

This package developed & and maintained by:
* [Martin Fleischmann](https://github.com/martinfleis)
* [Anastassia Vybornova](https://github.com/anastassiavybornova)
* [James D. Gaboardi](https://github.com/jGaboardi)

Copyright (c) 2024-, neatnet Developers
