__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpi4py import MPI
from cbcdns import config
import sys, cProfile
from numpy import *
from mpiFFT4py import slab_FFT, pencil_FFT, line_FFT
from cbcdns.utilities import *
from cbcdns.h5io import *
from cbcdns.optimization import *

# Import problem specific methods and solver methods specific to either slab or pencil decomposition
from cbcdns.mpi import *
from cbcdns.maths import *

comm = MPI.COMM_WORLD
comm.barrier()
num_processes = comm.Get_size()
rank = comm.Get_rank()

# Set types based on configuration
float, complex, mpitype = {"single": (float32, complex64, MPI.F_FLOAT_COMPLEX),
                           "double": (float64, complex128, MPI.F_DOUBLE_COMPLEX)}[config.precision]

# Apply correct precision and set mesh size
dt = config.dt = float(config.dt)
nu = config.nu = float(config.nu)
#eta = config.eta = float(config.eta)
M = config.M = array([eval(str(f)) for f in config.M], dtype=int)  # Convert from possible commandline, which is parsed as strings
L = config.L = array([eval(str(f)) for f in config.L], dtype=float)
N = 2**M
dx = (L/N).astype(float)

if config.decomposition == 'slab':
    FFT = slab_FFT(N, L, MPI, config.precision)
    
elif config.decomposition == 'pencil':
    if config.Pencil_Alignment == 'X':
        FFT = pencil_FFT['X'](N, L, MPI, config.precision, config.P1)
    else:
        FFT = pencil_FFT['Y'](N, L, MPI, config.precision, config.P1)
        
elif config.decomposition == 'line':
    FFT = line_FFT(N, L, MPI, config.precision)

if config.make_profile: profiler = cProfile.Profile()

# Set up solver using either slab or pencil decomposition
vars().update(setup(**vars()))

def update(t, tstep, **kwargs):
    pass

def set_source(Source, **kwargs):
    Source[:] = 0
    return Source

