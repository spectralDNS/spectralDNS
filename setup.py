#!/usr/bin/env python
import os
import re
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from numpy import get_include

cwd = os.path.abspath(os.path.dirname(__file__))
cdir = os.path.join(cwd, "spectralDNS", "optimization")
sdir = os.path.join(cwd, "spectralDNS", "shen")

def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    devnull = open(os.devnull, "w")
    p = subprocess.Popen([compiler.compiler[0], '-E', '-'] + [flagname],
                         stdin=subprocess.PIPE, stdout=devnull, stderr=devnull,
                         shell=True)
    p.communicate("")
    return True if p.returncode == 0 else False

class build_ext_subclass(build_ext):
    def build_extensions(self):
        extra_compile_args = ['-g0']
        for c in ['-w', '-Ofast', '-ffast-math', '-march=native']:
            if has_flag(self.compiler, c):
                extra_compile_args.append(c)

        for e in self.extensions:
            e.extra_compile_args += extra_compile_args
            e.include_dirs.extend([get_include()])
        build_ext.build_extensions(self)

def get_extension():
    ext = []
    for s in ("LUsolve", ):
        ext.append(Extension("spectralDNS.shen.{0}".format(s),
                             sources=[os.path.join(sdir, '{0}.pyx'.format(s))],
                             language="c++"))
    for e in ext:
        e.extra_link_args.extend(["-std=c++11"])

    for s in [files for files in os.listdir(cdir) if files.endswith('.pyx')]:
        ext.append(Extension("spectralDNS.optimization.{0}".format(s[:-4]),
                             sources=[os.path.join(cdir, '{0}'.format(s))],
                             language="c++"))

    try:
        from pythran import PythranExtension
        ext.append(PythranExtension('spectralDNS.optimization.pythran_maths',
                                    sources=['spectralDNS/optimization/pythran_maths.py']))
    except ImportError:
        print("Disabling Pythran support, package not available")

    return ext

def version():
    srcdir = os.path.join(cwd, 'spectralDNS')
    with open(os.path.join(srcdir, '__init__.py')) as f:
        m = re.search(r"__version__\s*=\s*'(.*)'", f.read())
        return m.groups()[0]

if __name__ == '__main__':
    setup(name="spectralDNS",
          version=version(),
          description="spectralDNS -- Spectral Navier-Stokes (and similar) solvers framework",
          long_description="",
          author="Mikael Mortensen",
          author_email="mikaem@math.uio.no",
          url='https://github.com/spectralDNS/spectralDNS',
          classifiers=[
              'Development Status :: 5 - Production/Stable',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Education',
              'Programming Language :: Python',
              'Programming Language :: Python :: 2',
              'Programming Language :: Python :: 3',
              'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Software Development :: Libraries :: Python Modules',
              ],
          install_requires=['numpy', 'shenfun', 'mpi4py-fft', 'mpi4py'],
          setup_requires=['numpy>=1.11', 'cython>=0.25', 'setuptools>=18.0'],
          packages=["spectralDNS",
                    "spectralDNS.h5io",
                    "spectralDNS.utilities",
                    "spectralDNS.maths",
                    "spectralDNS.shen",
                    "spectralDNS.solvers",
                    "spectralDNS.optimization",
                    ],
          package_dir={"spectralDNS": "spectralDNS"},
          ext_modules=get_extension(),
          cmdclass={'build_ext': build_ext_subclass}
         )
