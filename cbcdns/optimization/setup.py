"""To use optimization do:

python setup.py build_ext --inplace

in this folder. (Will need to work on improved installation routines)

"""
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

print sys.argv

if '--with-cython' in sys.argv[-1]:
    use_cython = True
    from Cython.Distutils import build_ext
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


if '--with-weave' in sys.argv[-1]:
    import weave_module
    # Compile weave extension modules for single and double precision
    mod1 = weave_module.weave_module("single")
    #mod1.compile(extra_compile_args=['-Ofast'], verbose=2)
    mod1.generate_file("weave_single.cpp")
    mod2 = weave_module.weave_module("double")
    #mod2.compile(extra_compile_args=['-Ofast'], verbose=2)
    mod2.generate_file("weave_double.cpp")

prec = {"single": ("float32", "complex64"),
        "double": ("float64", "complex128")}

ff = open("numba_module.py").read()
fs = open("numba_single.py", "w")
fs.write(ff.format(*prec["single"]))
fs.close()

fd = open("numba_double.py", "w")
fd.write(ff.format(*prec["double"]))
fd.close()
