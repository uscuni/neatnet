# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

import sphinx_autosummary_accessors

sys.path.insert(0, os.path.abspath("../neatnet/"))

import neatnet  # noqa

project = "neatnet"
copyright = "2024-, neatnet Developers"
author = "Martin Fleischmann, Anastassia Vybornova, James D. Gaboardi"

version = neatnet.__version__
release = neatnet.__version__

language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "numpydoc",
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinx_autosummary_accessors",
    "sphinx_copybutton",
]

bibtex_bibfiles = ["_static/references.bib"]

master_doc = "index"

templates_path = [
    "_templates",
    sphinx_autosummary_accessors.templates_path,
]
exclude_patterns = []

intersphinx_mapping = {
    "esda": (
        "https://pysal.org/esda/",
        "https://pysal.org/esda//objects.inv",
    ),
    "geopandas": ("https://geopandas.org/en/latest", None),
    "libpysal": (
        "https://pysal.org/libpysal/",
        "https://pysal.org/libpysal//objects.inv",
    ),
    "momepy": ("http://docs.momepy.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "shapely": ("https://shapely.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_use_plots = True
class_members_toctree = True
numpydoc_show_inherited_class_members = True
numpydoc_xref_param_type = True
autodoc_default_options = {"members": True, "undoc-members": True}
plot_include_source = True

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# html_logo = "_static/logo.svg"
# html_favicon = "_static/icon.png"
html_theme_options = {
    "use_sidenotes": True,
}
nb_execution_mode = "off"
autodoc_typehints = "none"
