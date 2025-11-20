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

   neatify

The minimal topology fixing can be done using another routine:

.. autosummary::
   :toctree: generated/

   fix_topology


Node Simplification
-------------------

Some of the individual components are also exposed as independent functions (note that
most are consumed by :func:`neatify`).


A subset of functions dealing with network nodes:

.. autosummary::
   :toctree: generated/

   consolidate_nodes
   remove_interstitial_nodes
   induce_nodes

Face artifact detection
-----------------------

A subset dealing with face artifacts:

.. autosummary::
   :toctree: generated/

   FaceArtifacts
   get_artifacts

Gap filling
-----------

Snapping and extending lines in case of imprecise topology:

.. autosummary::
   :toctree: generated/

   close_gaps
   extend_lines

Internal components
-------------------

For debugging purposes, users may use some parts of the internal API used within :func:`neatify`.

.. autosummary::
   :toctree: generated/

   get_artifacts
   neatify_loop
   neatify_singletons
   neatify_pairs
   neatify_clusters

None of the other functions is intended for public use and their API can change without a warning.