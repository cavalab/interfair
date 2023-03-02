# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Interfair'
copyright = '2023, William La Cava and Elle Lett'
author = 'William La Cava and Elle Lett'
release = ''

# add path to sys to import functions
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0,os.path.abspath(os.path.join(dir_path, '..')))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'numpydoc',
    # Sphinx's own extensions
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_inline_tabs",
    'nbsphinx'
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_material'
html_static_path = ['_static']

source_suffix = ['.rst', '.md']


html_theme_options = {
    'base_url': 'http://cavalab.org/interfair',
    'repo_url': 'https://github.com/cavalab/interfair/',
    'repo_name': 'Interfair',
    'html_minify': True,
    'css_minify': True,
    'nav_title': 'Interfair: Fomo',
    'logo_icon': '&#xe869',
    'globaltoc_depth': 2,
    "color_primary": "deep purple",
    "color_accent": "white",

}

html_sidebars = {
            "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
            }
