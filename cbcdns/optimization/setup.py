"""
This setup file is called by the top setup and is used to generate 
files with different precision
"""
import sys

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
