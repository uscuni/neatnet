.. _reference:

.. currentmodule:: neatnet

API reference
=============

The API reference provides an overview of all public functions in ``neatnet``.

Network Simplification Routines
-------------------------------

The top-level function that performs complete adaptive simplification of street networks
is the primary API of ``neatnet``.

.. autosummary::
   :toctree: generated/

   simplify_network

Simplification Components
~~~~~~~~~~~~~~~~~~~~~~~~~

Some of the individual components are also exposed as independent functions (note that
all are consumed by :func:`simplify_network`).

Either as combined routines:

.. autosummary::
   :toctree: generated/

   consolidate_nodes
   fix_topology

Or as their atomic components:

.. autosummary::
   :toctree: generated/

   remove_false_nodes
   induce_nodes
   FaceArtifacts


Additional functions
--------------------

Some of the functions for specific pre-processing tasks that are not part of the simplification routine:

.. autosummary::
   :toctree: generated/

   close_gaps
   extend_lines

Internal components
-------------------

For debugging purposes, users may use some parts of the internal API.

.. autosummary::
   :toctree: generated/

   get_artifacts
   simplify_loop
   simplify_singletons
   simplify_pairs
   simplify_clusters

None of the other functions is intended for public use and their API can change without a warning.