# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'NUCS'
copyright = '2024, Yan Georget'
author = 'Yan Georget'

release = '3.0.0'
version = '3.0.0'

# -- General configuration

tls_verify = False

extensions = [
    'sphinx.ext.linkcode',
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

def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if info['module']:
        filename = info['module'].replace('.', '/')
        return f"https://github.com/yangeorget/nucs/tree/main/{filename}.py"
    return None