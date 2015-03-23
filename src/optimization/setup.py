from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
import weave_ext

# Compile weave extension modules for single and double precision
weave_ext.weave_module("single")
weave_ext.weave_module("double")

# Compile cython extension modules for single and double precision
precision = {
  "single": """
ctypedef np.complex64_t complex_t
ctypedef np.float32_t real_t
ctypedef np.int64_t int_t
""",
  "double": """
ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t
"""      
}

ff = open("cython_module.py").read()
fs = open("cython_single.pyx", "w")
fs.write(ff.format(precision["single"]))
fs.close()

fd = open("cython_double.pyx", "w")
fd.write(ff.format(precision["double"]))
fd.close()

ext = Extension("cython_single", ["cython_single.pyx"],
                include_dirs = [get_include()])

ext2 = Extension("cython_double", ["cython_double.pyx"],
                 include_dirs = [get_include()])                

setup(ext_modules=[ext, ext2],
      cmdclass = {'build_ext': build_ext})
