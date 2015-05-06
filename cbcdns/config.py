__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-08"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Global run-time configuration that may be overloaded on the commandline
"""
import argparse
from numpy import pi

parser = argparse.ArgumentParser(prog='cbcdns')

parser.add_argument('solver', choices=('NS', 'VV', 'NS2D', 'MHD', 'Bq2D'), 
                    help="""Choose solver. NS is a regular velocity-pressure formulation and VV uses a velocity-vorticity formulation. NS2D is a regular 2D solver. MHD is a 3D MagnetoHydroDynamics solver.""")
parser.add_argument('--decomposition', default='slab', choices=('slab', 'pencil', 'line'), help="Choose 3D decomposition between slab and pencil. For 2D problems line is the only choice and is made automatically.")
parser.add_argument('--precision', default='double', choices=('single', 'double'))
parser.add_argument('--optimization', default='', choices=('cython', 'weave', 'numba', 'numexpr'))
parser.add_argument('--communication', default='alltoall', choices=('alltoall', 'sendrecv_replace'), help='only for slab')
parser.add_argument('--convection', default='Vortex', choices=('Standard', 'Divergence', 'Skewed', 'Vortex'))
parser.add_argument('--integrator', default='RK4', choices=('RK4', 'ForwardEuler', 'AB2'))
parser.add_argument('--make_profile', default=0, help='Enable cProfile profiler')
parser.add_argument('--nu', default=0.000625, type=float, help='Viscosity')
parser.add_argument('--eta', default=0.01, type=float, help='MHD parameter')
parser.add_argument('--dt', default=0.01, type=float, help='Time step size')
parser.add_argument('--T', default=0.1, type=float, help='End time')
parser.add_argument('--M', default=[6, 6, 6], nargs='+', help='Mesh size is pow(2, M[i]) in direction i')
parser.add_argument('--L', default=[2*pi, 2*pi, 2*pi], nargs='+', help='Physical mesh size')
parser.add_argument('--P1', default=1, type=int, help='pencil decomposition in first direction')
parser.add_argument('--write_result', default=1e8, type=int, help="Write results to HDF5 every...")
parser.add_argument('--write_yz_slice',  default=[0, 1e8], help="Write 2D slice to HDF5 [x index, every]")

def update(new):
    assert isinstance(new, dict)
    parser.set_defaults(**new) # allows extra arguments
