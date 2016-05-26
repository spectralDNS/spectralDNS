__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpi4py import MPI
import sys, cProfile
from numpy import *
from mpiFFT4py import slab_FFT, pencil_FFT, line_FFT, empty, zeros    # possibly byte-aligned zeros/empty
from spectralDNS.utilities import *
from spectralDNS.h5io import *
from spectralDNS.optimization import *
from spectralDNS.maths import *

comm = MPI.COMM_WORLD
comm.barrier()
num_processes = comm.Get_size()
rank = comm.Get_rank()

def get_FFT(N, L, decomposition, precision="double", P1=None, alignment="Y"):
    if decomposition == 'slab':
        assert len(N) == 3
        assert len(L) == 3
        FFT = slab_FFT(N, L, MPI, precision)
        
    elif decomposition == 'pencil':
        assert len(N) == 3
        assert len(L) == 3
        FFT = pencil_FFT(N, L, MPI, precision, P1=P1, alignment=alignment)
            
    elif decomposition == 'line':
        assert len(N) == 2
        assert len(L) == 2
        FFT = line_FFT(N, L, MPI, precision)
    return FFT

def update(**kwargs):
    pass

def additional_callback(**kw):
    pass

def set_source(Source, **kw):
    Source[:] = 0
    return Source
