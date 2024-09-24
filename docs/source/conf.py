# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'NUCS'
copyright = '2024, Yan Georget'
author = 'Yan Georget'

release = '0.9'
version = '0.9.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "yangeorget", # Username
    "github_repo": "nucs", # Repo name
    "github_version": "main", # Version
    "conf_py_path": "/docs/source/", # Path in the checkout to the docs root
}

# -- Options for EPUB output
epub_show_urls = 'footnote'