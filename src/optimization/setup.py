from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
import weave_module

# Compile weave extension modules for single and double precision
weave_module.weave_module("single")
weave_module.weave_module("double")

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
                include_dirs = [get_include()],
                extra_compile_args=["-Ofast"])

ext2 = Extension("cython_double", ["cython_double.pyx"],
                 include_dirs = [get_include()],
                 extra_compile_args=["-Ofast"])  

setup(ext_modules=[ext, ext2],
      cmdclass = {'build_ext': build_ext})

prec = {"single": ("float32", "complex64"),
        "double": ("float64", "complex128")}

ff = open("numba_module.py").read()
fs = open("numba_single.py", "w")
fs.write(ff.format(*prec["single"]))
fs.close()

fd = open("numba_double.py", "w")
fd.write(ff.format(*prec["double"]))
fd.close()
