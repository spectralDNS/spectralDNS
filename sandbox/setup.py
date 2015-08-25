from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy import get_include

#ext = cythonize(['TDMA.pyx', 'Cheb.pyx'])

ext = cythonize(Extension("TDMA", sources = ['TDMA.pyx'], language="c++"))
ext += cythonize(Extension("Cheb", sources = ['Cheb.pyx']))
ext += cythonize(Extension("Matvec", sources = ['Matvec.pyx']))

ext[0].extra_compile_args.extend(["-O3"])  
ext[1].extra_compile_args.extend(["-O3"])  
ext[2].extra_compile_args.extend(["-O3"])  
setup(
    name = "SFTc",
    ext_modules = ext,
    py_modules = ["SFTc.py"],
    cmdclass = {'build_ext': build_ext})
