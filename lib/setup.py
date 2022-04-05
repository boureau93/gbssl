from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = Extension("gbssl",sources=["nodes.c","pairs.c","graph.c","potts.c",
    "ssl.c","interface.pyx"],
    libraries=["gsl","gslcblas"],
    extra_compile_args=["-fopenmp", "-I/usr/include/gsl -lgsl -lgslblasnative"],
    extra_link_args=["-fopenmp", "-I/usr/include/gsl -lgsl -lgslblasnative"],
    include_dirs=[np.get_include()],
    language="c")

setup(ext_modules=cythonize(ext))