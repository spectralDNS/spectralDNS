"""
This setup file is called by the top setup and is used to generate 
files with different precision
"""
import sys

# Compile cython extension modules for single and double precision
precision = {
"single": """
ctypedef np.complex64_t complex_t
ctypedef np.float32_t real_t
ctypedef np.int64_t int_t
ctypedef float real
""",
"double": """
ctypedef np.complex128_t complex_t
ctypedef np.float64_t real_t
ctypedef np.int64_t int_t
ctypedef double real
"""      
}

for module in ("TDMA", "Cheb", "LUsolve", "Matvec"):
    ff = open("{0}_module.py".format(module)).read()
    fs = open("{0}_single.pyx".format(module), "w")
    fs.write(ff.format(precision["single"]))
    fs.close()

    fd = open("{0}_double.pyx".format(module), "w")
    fd.write(ff.format(precision["double"]))
    fd.close()
