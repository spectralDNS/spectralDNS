__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-08"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Global run-time configuration that may be overloaded on the commandline
"""
import argparse
from numpy import pi
import copy

parser = argparse.ArgumentParser(prog='spectralDNS', add_help=False)

parser.add_argument('--precision', default='double', choices=('single', 'double'))
parser.add_argument('--optimization', default='', choices=('cython', 'weave', 'numba'))
parser.add_argument('--communication', default='alltoall', choices=('alltoall', 'sendrecv_replace'), help='only for slab')
parser.add_argument('--make_profile', default=0, help='Enable cProfile profiler')
parser.add_argument('--dt', default=0.01, type=float, help='Time step size')
parser.add_argument('--T', default=0.1, type=float, help='End time')
parser.add_argument('--write_result', default=1e8, type=int, help='Write results to HDF5 every...')
parser.add_argument('--write_yz_slice',  default=[0, 1e8], help='Write 2D slice to HDF5 [x index, every]')
parser.add_argument('--checkpoint',  default=1e8, type=int, help='Save intermediate result every...')
parser.add_argument('--nu', default=0.000625, type=float, help='Viscosity')
parser.add_argument('--t', default=0.0, type=float, help='Time')
parser.add_argument('--tstep', default=0, type=int, help='Time step')
parser.add_argument("--time_integrator_tol",default=1.e-5,type=float,help="Tolerance for adaptive time-stepping")

# Arguments for isotropic DNS solver
triplyperiodic = argparse.ArgumentParser(parents=[parser])
triplyperiodic.add_argument('--convection', default='Vortex', choices=('Standard', 'Divergence', 'Skewed', 'Vortex'))
triplyperiodic.add_argument('--integrator', default='RK4', choices=('RK4', 'ForwardEuler', 'AB2',"BS5_adaptive","BS5_fixed","IMEX1","EXPBS5","IMEX4","EXPEULER","IMEX3","IMEX5"))
triplyperiodic.add_argument('--L', default=[2*pi, 2*pi, 2*pi], nargs=3, help='Physical mesh size')
triplyperiodic.add_argument('--dealias', default='2/3-rule', choices=('2/3-rule', '3/2-rule', 'None'), help='Choose dealiasing method')
triplyperiodic.add_argument('--Pencil_alignment', default='Y', choices=('X', 'Y'), help='Alignment of the complex data for pencil decomposition')
triplyperiodic.add_argument('--P1', default=1, type=int, help='pencil decomposition in first direction')
triplyperiodic.add_argument('--decomposition', default='slab', choices=('slab', 'pencil'), help="Choose 3D decomposition between slab and pencil.")
triplyperiodic.add_argument('--M', default=[6, 6, 6], nargs=3, help='Mesh size is pow(2, M[i]) in direction i')

trippelsubparsers = triplyperiodic.add_subparsers(dest='solver')

parser_NS = trippelsubparsers.add_parser('NS', help='Regular Navier Stokes solver')
parser_VV = trippelsubparsers.add_parser('VV', help='Velocity-Vorticity formulation')
parser_MHD = trippelsubparsers.add_parser('MHD', help='Magnetohydrodynamics solver')
parser_MHD.add_argument('--eta', default=0.01, type=float, help='MHD parameter')

doublyperiodic = argparse.ArgumentParser(parents=[parser])
doublyperiodic.add_argument('--integrator', default='RK4', choices=('RK4', 'ForwardEuler', 'AB2'))
doublyperiodic.add_argument('--L', default=[2*pi, 2*pi], nargs=2, help='Physical mesh size')
doublyperiodic.add_argument('--dealias', default='2/3-rule', choices=('2/3-rule', 'None'), help='Choose dealiasing method')
doublyperiodic.add_argument('--decomposition', default='line', choices=('line', ), help="For 2D problems line is the only choice.")
doublyperiodic.add_argument('--M', default=[6, 6], nargs=2, help='Mesh size is pow(2, M[i]) in direction i')

doublesubparsers = doublyperiodic.add_subparsers(dest='solver')

parser_NS2D = doublesubparsers.add_parser('NS2D', help='Regular 2D Navier Stokes solver')
parser_Bq2D = doublesubparsers.add_parser('Bq2D', help='Regular 2D Navier Stokes solver with Boussinesq model.')

# Arguments for Shen based solver with one inhomogeneous direction
channel = argparse.ArgumentParser(parents=[parser])
channel.add_argument('--convection', default='Vortex', choices=('Standard', 'Divergence', 'Skew', 'Vortex'))
channel.add_argument('--L', default=[2, 2*pi, 2*pi], nargs=3, help='Physical mesh size')
channel.add_argument('--dealias', default='3/2-rule', choices=('3/2-rule', '2/3-rule', 'None'), help='Choose dealiasing method')
channel.add_argument('--decomposition', default='slab', choices=('slab', 'pencil'), help="Choose 3D decomposition between slab and pencil.")
channel.add_argument('--Pencil_alignment', default='X', choices=('X',), help='Alignment of the complex data for pencil decomposition')
channel.add_argument('--M', default=[6, 6, 6], nargs=3, help='Mesh size is pow(2, M[i]) in direction i')

channelsubparsers = channel.add_subparsers(dest='solver')

KMM = channelsubparsers.add_parser('KMM', help='Kim Moin Moser channel solver with Crank-Nicolson and Adams-Bashforth discretization.')
KMMRK3 = channelsubparsers.add_parser('KMMRK3', help='Kim Moin Moser channel solver with third order semi-implicit Runge-Kutta discretization.')
IPCS = channelsubparsers.add_parser('IPCS', help='Incremental pressure correction with Crank-Nicolson and Adams-Bashforth discretization.')
IPCS.add_argument('--velocity_pressure_iters', default=1, type=int, help='Number of inner velocity pressure iterations for IPCS')
IPCS.add_argument('--print_divergence_progress', default=False, help='Print the norm of the pressure correction on inner iterations for IPCS')
IPCS.add_argument('--divergence_tol', default=1e-7, type=float, help='Tolerance on divergence error for pressure velocity coupling for IPCS')
IPCSR = channelsubparsers.add_parser('IPCSR', help='Incremental pressure correction with Crank-Nicolson and Adams-Bashforth discretization.')
IPCSR.add_argument('--velocity_pressure_iters', default=1, type=int, help='Number of inner velocity pressure iterations for IPCS')
IPCSR.add_argument('--print_divergence_progress', default=False, help='Print the norm of the pressure correction on inner iterations for IPCS')
IPCSR.add_argument('--divergence_tol', default=1e-7, type=float, help='Tolerance on divergence error for pressure velocity coupling for IPCS')


#from IPython import embed; embed()

## Arguments for Shen based solver for general bounday conditions
#ShenGeneralBCs = argparse.ArgumentParser(parents=[parser])
#ShenGeneralBCs.add_argument('--solver', default='IPCS_GeneralBCs', choices=('IPCS_GeneralBCs',), help="""Choose solver.""")
#ShenGeneralBCs.add_argument('--convection', default='Standard', choices=('Standard', 'Divergence'))
#ShenGeneralBCs.add_argument('--L', default=[2, 2*pi, 2*pi], nargs='+', help='Physical mesh size')
#ShenGeneralBCs.add_argument('--velocity_pressure_iters', default=1, type=int, help='Number of inner velocity pressure iterations')
#ShenGeneralBCs.add_argument('--print_divergence_progress', default=False, help='Print the norm of the pressure correction on inner iterations')
#ShenGeneralBCs.add_argument('--dealias', default='2/3-rule', choices=('3/2-rule', '2/3-rule', 'None'), help='Choose dealiasing method')
#ShenGeneralBCs.add_argument('--Pencil_alignment', default='X', choices=('X',), help='Alignment of the complex data for pencil decomposition')

## Arguments for Shen based solver for MHD
#ShenMHD = argparse.ArgumentParser(parents=[parser])
#ShenMHD.add_argument('--solver', default='IPCS_MHD', choices=('IPCS_MHD',), help='Choose solver')
#ShenMHD.add_argument('--convection', default='Standard', choices=('Standard', 'Divergence'))
#ShenMHD.add_argument('--L', default=[2, 2*pi, 2*pi], nargs='+', help='Physical mesh size')
#ShenMHD.add_argument('--eta', default=0.0016666, type=float, help='Resistivity')
#ShenMHD.add_argument('--Ha', default=0.0043817804600413289, type=float, help='Hartmann number')
#ShenMHD.add_argument('--B_strength', default=0.000001, type=float, help='Magnetic strength')
#ShenMHD.add_argument('--velocity_pressure_iters', default=1, type=int, help='Number of inner velocity pressure iterations')
#ShenMHD.add_argument('--print_divergence_progress', default=False, help='Print the norm of the pressure correction on inner iterations')
#ShenMHD.add_argument('--dealias', default='2/3-rule', choices=('3/2-rule', '2/3-rule', 'None'), help='Choose dealiasing method')
#ShenMHD.add_argument('--Pencil_alignment', default='X', choices=('X',), help='Alignment of the complex data for pencil decomposition')

def update(new, mesh="triplyperiodic"):
    assert isinstance(new, dict)
    exec mesh + ".set_defaults(**new)"

