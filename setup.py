#!/usr/bin/env python
import setuptools

import os, sys, platform
from distutils.core import setup, Extension
import subprocess
from numpy import get_include
from Cython.Distutils import build_ext
from Cython.Build import cythonize
try:
    from Cython.Compiler.Options import get_directive_defaults
    directive_defaults = get_directive_defaults()
    #directive_defaults['linetrace'] = True
    #directive_defaults['binding'] = True
except ImportError:
    pass

#define_macros=[('CYTHON_TRACE', '1')]
define_macros=None

# Version number
major = 1
minor = 0

cwd = os.path.abspath(os.path.dirname(__file__))
cdir = os.path.join(cwd, "spectralDNS", "optimization")
sdir = os.path.join(cwd, "spectralDNS", "shen")

ext = None
cmdclass = {}
class build_ext_subclass(build_ext):
    def build_extensions(self):
        extra_compile_args = ['-w', '-Ofast']
        devnull = open(os.devnull,"w")
        cmd = "%s -E - %s &>/dev/null" % (
            self.compiler.compiler[0], " ".join(extra_compile_args))
        p = subprocess.Popen([self.compiler.compiler[0], '-E', '-'] + extra_compile_args,
                             stdin=subprocess.PIPE, stdout=devnull, stderr=devnull)
        out = p.communicate("")
        if p.returncode != 0:
            extra_compile_args = ['-w', '-O3']
        for e in self.extensions:
            e.extra_compile_args += extra_compile_args
        build_ext.build_extensions(self)

args = ""
if not "sdist" in sys.argv:
    if "build_ext" in sys.argv:
        args = "build_ext --inplace"
    subprocess.call([sys.executable, os.path.join(cdir, "setup.py"),
                    args], cwd=cdir)

    ext = []
    for s in ("LUsolve", "Matvec"):
        ext += cythonize(Extension("spectralDNS.shen.{0}".format(s),
                                   sources=[os.path.join(sdir, '{0}.pyx'.format(s))],
                                   language="c++", define_macros=define_macros))
    [e.extra_link_args.extend(["-std=c++11"]) for e in ext]

    [e.include_dirs.extend([get_include()]) for e in ext]
    ext0 = []
    ff = [files for files in os.listdir(cdir) if files.endswith('.pyx')]
    for s in ff:
        ext0 += cythonize(Extension("spectralDNS.optimization.{0}".format(s[:-4]),
                                    sources=[os.path.join(cdir, '{0}'.format(s))],
                                    language="c++", define_macros=define_macros))
    [e.include_dirs.extend([get_include()]) for e in ext0]
    ext += ext0

    try:
        from pythran import PythranExtension
        ext.append(PythranExtension('spectralDNS.optimization.pythran_maths',
                                    sources=['spectralDNS/optimization/pythran_maths.py']))
    except ImportError:
        print("Disabling Pythran support, package not available")
        pass

    cmdclass = {'build_ext': build_ext_subclass}

else:
    # Remove generated files
    for name in os.listdir(cdir):
        if "single" in name or "double" in name:
            os.remove(os.path.join(cdir, name))

setup(name = "spectralDNS",
      version = "%d.%d" % (major, minor),
      description = "spectralDNS -- Spectral Navier-Stokes solvers framework",
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
      packages = ["spectralDNS",
                  "spectralDNS.h5io",
                  "spectralDNS.utilities",
                  "spectralDNS.maths",
                  "spectralDNS.shen",
                  "spectralDNS.solvers",
                  "spectralDNS.optimization",
                  ],
      package_dir = {"spectralDNS": "spectralDNS"},
      ext_modules = ext,
      cmdclass = cmdclass
    )
