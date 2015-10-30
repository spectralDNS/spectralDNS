#!/usr/bin/env python

import os, sys, platform
#from setuptools import setup, Extension
from distutils.core import setup, Extension
import subprocess
from numpy import get_include
 
args = ''
if '--with-cython' in sys.argv:
    sys.argv.remove('--with-cython')
    args += '--with-cython '
    use_cython = True
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize

else:
    use_cython = False

if '--with-weave' in sys.argv:
    sys.argv.remove('--with-weave')
    args += '--with-weave '
    use_weave = True
    from scipy import weave

else:
    use_weave = False
    
# Version number
major = 1
minor = 0
maintenance = 0

cwd = os.path.abspath(os.path.dirname(__file__))
cdir = os.path.join(cwd, "cbcdns", "optimization")
sdir = os.path.join(cwd, "cbcdns", "shen")

ext = None
cmdclass = {}
if not "sdist" in sys.argv:
    if "build_ext" in sys.argv:
        args += "build_ext --inplace"
    subprocess.call([sys.executable, os.path.join(cdir, "setup.py"),
                    args], cwd=cdir)
    subprocess.call([sys.executable, os.path.join(sdir, "setup.py"),
                    args], cwd=sdir)    
                    
    ext = []
    for prec in ("single", "double"):
        for s in ("LUsolve", "TDMA"):
            ext += cythonize(Extension("cbcdns.shen.{}_{}".format(s, prec), sources = [os.path.join(sdir, '{}_{}.pyx'.format(s, prec))], language="c++"))
        
        for s in ("Cheb", "Matvec"):
            ext += cythonize(Extension("cbcdns.shen.{}_{}".format(s, prec), sources = [os.path.join(sdir, '{}_{}.pyx'.format(s, prec))]))
    
    
    [e.extra_compile_args.extend(["-Ofast"]) for e in ext]
    [e.include_dirs.extend([get_include()]) for e in ext]
    if use_cython:
        ext0 = cythonize(os.path.join(cdir, "*.pyx"))
        [e.extra_compile_args.extend(["-Ofast"]) for e in ext0]
        [e.include_dirs.extend([get_include()]) for e in ext0]
        ext += ext0
        cmdclass = {'build_ext': build_ext}
        
    if use_weave:
        weave_dir,junk = os.path.split(os.path.abspath(weave.__file__))
        includes = [get_include(), weave_dir]
        includes.append(os.path.join(weave_dir, 'blitz'))
        includes.append(os.path.join(weave_dir, 'scxx'))
        imp_source = os.path.join(weave_dir, 'scxx/weave_imp.cpp')
        ext += [Extension("cbcdns.optimization.weave_single", [os.path.join(cdir, "weave_single.cpp"), imp_source],
                         include_dirs=includes, extra_compile_args=["-Ofast"])]

        ext += [Extension("cbcdns.optimization.weave_double", [os.path.join(cdir, "weave_double.cpp"), imp_source],
                        include_dirs = includes, extra_compile_args=["-Ofast"])]
        cmdclass = {'build_ext': build_ext}
    
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
      url = 'https://github.com/mikaem/spectralDNS',
      zip_safe = False,
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
                  "cbcdns.mpi",
                  "cbcdns.fft",
                  "cbcdns.utilities",
                  "cbcdns.maths",
                  "cbcdns.solvers",
                  "cbcdns.optimization",
                  ],
      package_dir = {"cbcdns": "cbcdns"},
      ext_modules = ext,
      cmdclass = cmdclass
    )
