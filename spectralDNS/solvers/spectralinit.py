__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpi4py import MPI
from spectralDNS import config
import sys, cProfile
from numpy import *
from mpiFFT4py import slab_FFT, pencil_FFT, line_FFT
from spectralDNS.utilities import *
from spectralDNS.h5io import *
from spectralDNS.optimization import *

# Import problem specific methods and solver methods specific to either slab or pencil decomposition
from spectralDNS.mesh import *
from spectralDNS.maths import *

import spectralDNS.context

def set_source(Source, **kwargs):
    Source[:] = 0
    return Source

