spectralDNS
=======

[![Build Status](https://travis-ci.org/spectralDNS/spectralDNS.svg?branch=master)](https://travis-ci.org/spectralDNS/spectralDNS)
[![CircleCI](https://circleci.com/gh/spectralDNS/spectralDNS/tree/master.svg?style=svg)](https://circleci.com/gh/spectralDNS/spectralDNS/tree/master)

Description
----------

spectralDNS contains a classical high-performance pseudo-spectral Navier-Stokes DNS solver for triply periodic domains. The most notable feature of this solver is that it's written entirely in Python using NumPy, MPI for Python (mpi4py) and pyFFTW. MPI decomposition is performed using either the "slab" or the "pencil" approach and, stripping away unnecessary pre- and post-processing steps, the slab solver is no more than 100 lines long, including the MPI. The code has been found to scale very well in tests on the Shaheen Blue Gene/P supercomputer at KAUST Supercomputing Laboratory. Results of both weak and strong scaling tests are shown below. In addition to incompressible Navier-Stokes there are also solvers for MHD and Navier-Stokes or MHD with variable density through a Boussinesq approximation. The solver is described more thoroughly in this paper:

M. Mortensen and H. P. Langtangen "High performance Python for direct numerical simulations of turbulent flows", Computer Physics Communications 203, p 53-65 (2016) http://arxiv.org/pdf/1602.03638v1.pdf

The efficiency of the pure NumPy/mpi4py solver has been enhanced using Cython for certain routines. The strong scaling results on Shaheen shown below have used the optimized Python/Cython solver, which is found to be faster than a pure C++ implementation of the same solver.

A channel flow solver is implemented using the Shen basis (Jie Shen, SIAM Journal on Scientific Computing, 16, 74-87, 1995) for the scheme described by Kim, Moin and Moser (J. Fluid Mechanics, Vol 177, 133-166, 1987).

Installation
-----------
spectralDNS is installed by cloning or forking the repository and then with regular python distutils

    python setup.py install --prefix="path used for installation. Must be on the PYTHONPATH"
    
or in-place using

    python setup.py build_ext --inplace

spectralDNS requires that [mpiFFT4py](https://github.com/spectralDNS/mpiFFT4py) has been installed already. Other than that, it requires [*h5py*](http://www.h5py.org) built with parallel HDF5, for visualizing the results.  [*cython*](http://cython.org) is used to optimize a few routines. 

To install using Anaconda, you may compile it yourselves using (from the main directory)

    conda build -c conda-forge -c spectralDNS conf/conda
    conda install spectralDNS --use-local

which will also build and install mpiFFT4py.
You may also use precompiled binaries in the [*spectralDNS*](https://anaconda.org/spectralDNS/mpifft4py) channel on Anaconda cloud. Use for exampel

    conda create --name spectralDNS -c conda-forge -c spectralDNS spectralDNS
    source activate spectralDNS

which installs both spectralDNS, mpiFFT4py and all required dependencies, most of which are pulled in from the conda-forge channel. There are binaries compiled for both OSX and linux, for either Python version 2.7 or 3.4. To specify the Python version as 3.4 instead of default (used above) you can for exampel do

    conda create --name spectralDNS_py3 -c conda-forge -c spectralDNS python=3 spectralDNS
    source activate spectralDNS_py3

Usage
-----
See the demo folder for usage.

There are different solvers. For example, there are two Navier Stokes solvers for the triply periodic domain. A regular one (solvers/NS.py), and one based on a velocity-vorticity formulation (solvers/VV.py). The solver of your choice is required as an argument when running:
    
    cd demo
    mpirun -np 4 python TG.py NS
    
or

    mpirun -np 4 python TG.py VV
    
There are many different arguments to each solver. They are all described in config.py. Arguments may be specified on the commandline

    mpirun -np 4 python TG.py --M 6 6 6 --precision single --dealias '3/2-rule' NS
    
before the required solver argument. Alternatively, use config.update as shown in demo/TG.py.

Scaling
------
The most recent simulations of the pencil version of the NS solver are showing excellent scaling up to 65k cores at KAUST and Shaheen II!

<p align="center">
    <img src="https://www.dropbox.com/s/thkaty8ow6m5xgh/strong_scaling_pencil_col.png?dl=1" width="600" height="400" alt="Strong scaling of Cython optimized NS solver on Shaheen II"/>
</p>
The difference between the red triangles and the blue squares are simply the number of mpi processes sent to each node on Shaheen II. The default version (blue squares) are filling up all nodes with 32 processes on each, whereas the red triangles have only four processes for each node. The green dots have been sampled using the default settings.

Results also from the old Shaheen BlueGene/P:

<p align="center">
    <img src="https://www.dropbox.com/s/pi4f25c0pyluxz0/weak_scaling_shaheen_numpy_noopt.png?dl=1" width="600" height="400" alt="Weak scaling of pure numpy/mpi4py solver on Shaheen BlueGene/P"/>
</p>
<p align="center">
    Weak scaling of pure numpy/mpi4py solver on Shaheen BlueGene/P. The C++ solver uses slab decomposition and MPI communication is performed by the FFTW library.
</p>

<p align="center">
    <img src="https://www.dropbox.com/s/p7uapi7eaqjmham/strong_scaling_shaheen_512.png?dl=1" width="600" height="400" alt="Strong scaling of optimized Python/Cython solver on Shaheen BlueGene/P"/>
</p>
<p align="center">
    Strong scaling of optimized Python/Cython solver on Shaheen BlueGene/P. The C++ solver uses slab decomposition and MPI communication is performed by the FFTW library.
</p>

<p align="center">
    <img src="https://www.dropbox.com/s/fsk7xolb2yk60re/Re2000_5.gif?dl=1" width="800" height="266" alt="Channel flow"/>
</p>
<p align="center">
    Turbulent channel flow at Re_tau = 2000. Simulations are performed using 512 cores on Shaheen II with the KMM channel flow solver (solvers/ShenKMM.py) using 512x1024x1024 points in real physical space.
</p>

<p align="center">
    <img src="https://www.dropbox.com/s/8oayxts0ix359hi/KHmovie1.gif?dl=1" width="600" height="400" alt="Kelvin Helmholtz instability"/>
</p>

<p align="center">
    Evolution of vorticity. Two-dimensional simulation of Kelvin Helmholtz shear instability using a Boussinesq formulation (solvers/spectralDNS2D_Boussinesq.py)
</p>

Authors
-------
spectralDNS is developed by

  * Mikael Mortensen
  * Diako Darian

Licence
-------
spectralDNS is licensed under the GNU GPL, version 3 or (at your option) any later version. spectralDNS is Copyright (2014-2016) by the authors.

Contact
-------
The latest version of this software can be obtained from

  https://github.com/spectralDNS/spectralDNS

Please report bugs and other issues through the issue tracker at:

  https://github.com/spectralDNS/spectralDNS/issues
