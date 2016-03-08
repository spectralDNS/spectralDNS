#!/usr/bin/env python

import os, sys, platform
#from setuptools import setup, Extension
from distutils.core import setup, Extension
import subprocess
from numpy import get_include
from Cython.Distutils import build_ext
from Cython.Build import cythonize
 
    
# Version number
major = 1
minor = 0
maintenance = 0

cwd = os.path.abspath(os.path.dirname(__file__))
cdir = os.path.join(cwd, "cbcdns", "optimization")
sdir = os.path.join(cwd, "cbcdns", "shen")
sgdir = os.path.join(cwd, "cbcdns", "shenGeneralBCs")

ext = None
cmdclass = {}
class build_ext_subclass(build_ext):
    def build_extensions(self):
        extra_compile_args = ['-w', '-Ofast']
        cmd = "echo | %s -E - %s &>/dev/null" % (
            self.compiler.compiler[0], " ".join(extra_compile_args))
        try:
            subprocess.check_call(cmd, shell=True)
        except:
            extra_compile_args = ['-w', '-O3']
        for e in self.extensions:
            e.extra_compile_args = extra_compile_args
        build_ext.build_extensions(self)

args = ""
if not "sdist" in sys.argv:
    if "build_ext" in sys.argv:
        args = "build_ext --inplace"
    subprocess.call([sys.executable, os.path.join(cdir, "setup.py"),
                    args], cwd=cdir)
    
    ext = []
    for s in ("LUsolve", "TDMA", "PDMA", "Matvec"):
        ext += cythonize(Extension("cbcdns.shen.{0}".format(s), sources = [os.path.join(sdir, '{0}.pyx'.format(s))], language="c++"))
        
    for s in ("Cheb", "HelmholtzMHD"):
        ext += cythonize(Extension("cbcdns.shen.{0}".format(s), sources = [os.path.join(sdir, '{0}.pyx'.format(s))]))    
    
    for s in ("LUsolve", "TDMA", "PDMA", "UTDMA"):
        ext += cythonize(Extension("cbcdns.shenGeneralBCs.{0}".format(s), sources = [os.path.join(sgdir, '{0}.pyx'.format(s))], language="c++"))
        
    for s in ("Matvec", "Matrices"):
        ext += cythonize(Extension("cbcdns.shenGeneralBCs.{0}".format(s), sources = [os.path.join(sgdir, '{0}.pyx'.format(s))]))    

    #extra_compile_args = ['-w', '-Ofast']
    #[e.extra_compile_args.extend(extra_compile_args) for e in ext]
    [e.include_dirs.extend([get_include()]) for e in ext]
    ext0 = cythonize(os.path.join(cdir, "*.pyx"))
    #[e.extra_compile_args.extend(extra_compile_args) for e in ext0]
    [e.include_dirs.extend([get_include()]) for e in ext0]
    ext += ext0
    cmdclass = {'build_ext': build_ext_subclass}
            
else:
    # Remove generated files
    for name in os.listdir(cdir):
        if "single" in name or "double" in name:
            os.remove(os.path.join(cdir, name))
 
setup(name = "cbcdns",
      version = "%d.%d.%d" % (major, minor, maintenance),
      description = "cbcdns -- Spectral Navier-Stokes solvers framework from the Center of Biomedical Computing",
      long_description = "",
      author = "Mikael Mortensen",
      author_email = "mikaem@math.uio.no", 
      url = 'https://github.com/spectralDNS/spectralDNS',
      classifiers = [
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Programming Language :: Python :: 2.7',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages = ["cbcdns",
                  "cbcdns.h5io",
                  "cbcdns.mesh",
                  "cbcdns.utilities",
                  "cbcdns.maths",
                  "cbcdns.shen",
                  "cbcdns.shenGeneralBCs",
                  "cbcdns.solvers",
                  "cbcdns.optimization",
                  ],
      package_dir = {"cbcdns": "cbcdns"},
      ext_modules = ext,
      cmdclass = cmdclass
    )
