"""
This setup file is called by the top setup and is used to generate
files with different precision
"""
import os

# Note to self. May be solved more elegantly using fused types,
# but this leads to much heavier modules (templates) that are slower
# in tests
#ctypedef np.int64_t int_t
#ctypedef fused complex_t:
    #np.complex128_t
    #np.complex64_t

#ctypedef fused real_t:
    #np.float64_t
    #np.float32_t

#ctypedef fused T:
    #np.float64_t
    #np.float32_t
    #np.int64_t

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

for module in ("integrators", "maths", "solvers"):
    # Use timestamp to determine if recompilation is required
    t0 = os.path.getmtime("cython_{0}.in".format(module))
    compile_new = False
    if not os.path.exists("cython_single_{0}.pyx".format(module)):
        compile_new = True
    elif os.path.getmtime("cython_single_{0}.pyx".format(module)) < t0:
        compile_new = True
    if compile_new:
        ff = open("cython_{0}.in".format(module)).read()
        fs = open("cython_single_{0}.pyx".format(module), "w")
        fs.write(ff.format(precision["single"]))
        fs.close()
        fd = open("cython_double_{0}.pyx".format(module), "w")
        fd.write(ff.format(precision["double"]))
        fd.close()

prec = {"single": ("float32", "complex64"),
        "double": ("float64", "complex128")}

t0 = os.path.getmtime("numba_module.in")
compile_new = False
if not os.path.exists("numba_module.py"):
    compile_new = True
elif os.path.getmtime("numba_module.py") < t0:
    compile_new = True
if compile_new:
    ff = open("numba_module.in").read()
    fs = open("numba_single.py", "w")
    fs.write(ff.format(*prec["single"]))
    fs.close()

    fd = open("numba_double.py", "w")
    fd.write(ff.format(*prec["double"]))
    fd.close()
