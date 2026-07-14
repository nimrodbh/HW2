from setuptools import setup, Extension

# Define the C extension module
module = Extension(
    'mykmeanssp', # The name of the module as it will be imported in Python
    sources=['kmeansmodule.c'] # The C source file(s)
)

# Setup script to build the extension
setup(
    name='mykmeanssp',
    version='1.0',
    description='K-means clustering C extension for Python',
    ext_modules=[module]
)
