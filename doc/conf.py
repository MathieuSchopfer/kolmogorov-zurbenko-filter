# -*- coding: utf-8 -*-

project = "Kolmogorov-Zurkbenko filter"
f_prefix = "kolzu_filter"
author = 'Mathieu Schopfer'
copyright = author + ' 2017'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

intersphinx_mapping = {
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'python': ('http://docs.python.org/3', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
}

# The master toctree document.
master_doc = 'index'

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinxdoc'
# html_theme = 'alabaster'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
