from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
"""
#this one has a bug:

setup(
    ext_modules=[
        Extension("ShellVolumeMasked", ["ShellVolumeMasked.c"],
                  include_dirs=[np.get_include()]),
    ],
)



# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

#this one has a bug:

setup(
    ext_modules=cythonize("ShellVolumeMasked.pyx"),
    include_dirs=[np.get_include()]
) 
"""
ext = Extension(name="ShellVolumeMasked",sources=["ShellVolumeMasked.pyx"],include_dirs=[np.get_include()])
setup(ext_modules=cythonize(ext))

# run in terminal:
# python setup.py build_ext --inplace
# or
# python setup.py develop --user