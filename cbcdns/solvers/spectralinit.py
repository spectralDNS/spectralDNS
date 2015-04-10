__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from mpi4py import MPI
from cbcdns import config
import sys, cProfile
from numpy import *
from cbcdns.utilities import *
from cbcdns.h5io import *
from cbcdns.optimization import *

# Parse parameters from the command line and update config
#commandline_kwargs = parse_command_line(sys.argv[1:])
#config.update(commandline_kwargs)

# Import problem specific methods and solver methods specific to either slab or pencil decomposition
from cbcdns.mpi import setup, ifftn_mpi, fftn_mpi
from cbcdns.maths import *

comm = MPI.COMM_WORLD
comm.barrier()
num_processes = comm.Get_size()
rank = comm.Get_rank()

# Set types based on configuration
float, complex, mpitype = {"single": (float32, complex64, MPI.F_FLOAT_COMPLEX),
                           "double": (float64, complex128, MPI.F_DOUBLE_COMPLEX)}[config.precision]

# Apply correct precision and set mesh size
dt = float(config.dt)
nu = float(config.nu)
N = 2**config.M
L = float(2*pi)
dx = float(L/N)

hdf5file = HDF5Writer(comm, dt, N, vars(config), float)
if config.make_profile: profiler = cProfile.Profile()

# Set up solver using wither slab or decomposition
vars().update(setup(**vars()))

def update(**kwargs):
    pass

def initialize(**kwargs):
    pass

def set_source(Source, **kwargs):
    Source[:] = 0
    return Source
