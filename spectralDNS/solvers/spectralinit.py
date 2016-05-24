__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpi4py import MPI
from spectralDNS import config
import sys, cProfile
from numpy import *
from mpiFFT4py import slab_FFT, pencil_FFT, line_FFT, empty, zeros  # possibly byte-aligned zeros/empty
from spectralDNS.utilities import *
from spectralDNS.h5io import *
from spectralDNS.optimization import *
from spectralDNS.maths import *

comm = MPI.COMM_WORLD
comm.barrier()
num_processes = comm.Get_size()
rank = comm.Get_rank()

# Set types based on configuration
float, complex, mpitype = {"single": (float32, complex64, MPI.F_FLOAT_COMPLEX),
                           "double": (float64, complex128, MPI.F_DOUBLE_COMPLEX)}[config.precision]

# Apply correct precision and set mesh size
config.dt = float(config.dt)
config.nu = float(config.nu)
M = config.M = array([eval(str(f)) for f in config.M], dtype=int)  # Convert from possible commandline, which is parsed as strings
L = config.L = array([eval(str(f)) for f in config.L], dtype=float)
N = 2**M
dx = (L/N).astype(float)

if config.mesh in ('doublyperiodic', 'triplyperiodic'):
    if config.decomposition == 'slab':
        FFT = slab_FFT(N, L, MPI, config.precision)
        
    elif config.decomposition == 'pencil':
        FFT = pencil_FFT(N, L, MPI, config.precision, P1=config.P1, alignment=config.Pencil_alignment)
            
    elif config.decomposition == 'line':
        FFT = line_FFT(N, L, MPI, config.precision)

if config.make_profile: profiler = cProfile.Profile()

def update(**kwargs):
    pass

def set_source(Source, **kwargs):
    Source[:] = 0
    return Source

