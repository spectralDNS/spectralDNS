"""
This setup file is called by the top setup and is used to generate 
files with different precision
"""
import sys

if '--with-cython' in sys.argv[-1]:
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

    for module in ("integrators", "maths", "solvers", "mpi"):
        ff = open("cython_{0}.py".format(module)).read()
        fs = open("cython_single_{0}.pyx".format(module), "w")
        fs.write(ff.format(precision["single"]))
        fs.close()

        fd = open("cython_double_{0}.pyx".format(module), "w")
        fd.write(ff.format(precision["double"]))
        fd.close()

if '--with-weave' in sys.argv[-1]:
    import weave_module
    # Compile weave extension modules for single and double precision
    mod1 = weave_module.weave_module("single")
    mod1.generate_file("weave_single.cpp")
    mod2 = weave_module.weave_module("double")
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
