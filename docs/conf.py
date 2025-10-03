# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'openghg_inversions'
copyright = '2025, Eric Saboya and Brendan Murphy'
author = 'Eric Saboya and Brendan Murphy'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pytensor": ("https://pytensor.readthedocs.io/en/latest/", None),
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
    "openghg": ("https://docs.openghg.org", None)
}

autosectionlabel_prefix_document = True

# napoleon settings (for google docstring style)
napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True


# Mock heavy or optional imports to prevent autodoc import failures
autodoc_mock_imports = ["cartopy"]

# Optionally, ignore missing references for certain types
nitpicky = False  # TODO: set to True once docs working
nitpick_ignore = [
    ('py:class', 'optional'),
    ('py:class', 'list'),
    ('py:class', 'dict'),
    ('py:class', 'OrderedDict'),
    ('py:class', 'dictionary'),
    ('py:class', 'function'),
    ('py:class', 'output'),
    ('py:class', 'outputname'),
    ('py:class', 'lists'),
    ('py:class', 'area'),
    ('py:class', 'version'),
    # Add others as needed
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
