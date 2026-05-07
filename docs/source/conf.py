# Configuration file for the Sphinx documentation builder.
import tomllib
from pathlib import Path

# -- Project information

project = 'NuCS'
copyright = '2024-2026, Yan Georget'
author = 'Yan Georget'

with open(Path(__file__).parent.parent.parent / "pyproject.toml", "rb") as f:
    data = tomllib.load(f)
release = data["project"]["version"]
version = ".".join(release.split(".")[:2])

# -- General configuration

tls_verify = False

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

autoclass_content = "class"

# -- Options for HTML output

html_theme = 'alabaster'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "yangeorget",  # Username
    "github_repo": "nucs",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/source/",  # Path in the checkout to the docs root
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


def _strip_signature(app, what, name, obj, options, signature, return_annotation):
    if what in ("function", "method"):
        return ("(...)", return_annotation)
    if what in ("class"):
        return ("", return_annotation)


def setup(app):
    app.connect("autodoc-process-signature", _strip_signature)
