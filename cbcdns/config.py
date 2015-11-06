__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-08"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Global run-time configuration that may be overloaded on the commandline
"""
import argparse
from numpy import pi

parser = argparse.ArgumentParser(prog='cbcdns', add_help=False)

parser.add_argument('--decomposition', default='slab', choices=('slab', 'pencil', 'line'), help="Choose 3D decomposition between slab and pencil. For 2D problems line is the only choice and is made automatically.")
parser.add_argument('--precision', default='double', choices=('single', 'double'))
parser.add_argument('--optimization', default='', choices=('cython', 'weave', 'numba', 'numexpr'))
parser.add_argument('--communication', default='alltoall', choices=('alltoall', 'sendrecv_replace'), help='only for slab')
parser.add_argument('--make_profile', default=0, help='Enable cProfile profiler')
parser.add_argument('--dt', default=0.01, type=float, help='Time step size')
parser.add_argument('--T', default=0.1, type=float, help='End time')
parser.add_argument('--M', default=[6, 6, 6], nargs='+', help='Mesh size is pow(2, M[i]) in direction i')
parser.add_argument('--P1', default=1, type=int, help='pencil decomposition in first direction')
parser.add_argument('--write_result', default=1e8, type=int, help="Write results to HDF5 every...")
parser.add_argument('--write_yz_slice',  default=[0, 1e8], help="Write 2D slice to HDF5 [x index, every]")
parser.add_argument('--checkpoint',  default=1e8, type=int, help="Save intermediate result every...")
parser.add_argument('--nu', default=0.000625, type=float, help='Viscosity')
parser.add_argument('--t', default=0.0, type=float, help='Time')
parser.add_argument('--tstep', default=0, type=int, help='Time step')

# Arguments for isotropic DNS solver
Isotropic = argparse.ArgumentParser(parents=[parser])
Isotropic.add_argument('--solver', default='NS', choices=('NS', 'VV', 'NS2D', 'MHD', 'Bq2D'), 
                    help="""Choose solver. NS is a regular velocity-pressure formulation and VV uses a velocity-vorticity formulation. NS2D is a regular 2D solver. MHD is a 3D MagnetoHydroDynamics solver.""")
Isotropic.add_argument('--convection', default='Vortex', choices=('Standard', 'Divergence', 'Skewed', 'Vortex'))
Isotropic.add_argument('--integrator', default='RK4', choices=('RK4', 'ForwardEuler', 'AB2'))
Isotropic.add_argument('--eta', default=0.01, type=float, help='MHD parameter')
Isotropic.add_argument('--L', default=[2*pi, 2*pi, 2*pi], nargs='+', help='Physical mesh size')

# Arguments for Shen based solver with one inhomogeneous direction
Shen = argparse.ArgumentParser(parents=[parser])
Shen.add_argument('--solver', default='IPCS', choices=('IPCS',), help="""Choose solver.""")
Shen.add_argument('--convection', default='Standard', choices=('Standard', 'Divergence'))
Shen.add_argument('--L', default=[2, 2*pi, 2*pi], nargs='+', help='Physical mesh size')
Shen.add_argument('--velocity_pressure_iters', default=1, type=int, help="Number of inner velocity pressure iterations")
Shen.add_argument('--print_divergence_progress', default=False, help="Print the norm of the pressure correction on inner iterations")

def update(new, par="Isotropic"):
    assert isinstance(new, dict)
    if par == "Isotropic":
        Isotropic.set_defaults(**new) # allows extra arguments
    elif par == "Shen":
        Shen.set_defaults(**new)
    