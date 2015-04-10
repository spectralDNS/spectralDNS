"""
Global run-time configuration that may be overloaded on the commandline
"""
import argparse
parser = argparse.ArgumentParser(prog='spectral.py')
parser.add_argument('--solver', default='NS', choices=('NS', 'VV', 'NS2D'), help="Choose solver between NS and VV, where NS uses a regular velocity-pressure formulation and VV uses a velocity-vorticity formulation.")
parser.add_argument('--decomposition', default='slab', choices=('slab', 'pencil'))
parser.add_argument('--precision', default='double', choices=('single', 'double'))
parser.add_argument('--optimization', default='', choices=('cython', 'weave', 'numba', 'numexpr'))
parser.add_argument('--communication', default='alltoall', choices=('alltoall', 'sendrecv_replace'), help='only for slab')
parser.add_argument('--convection', default='Vortex', choices=('Standard', 'Divergence', 'Skewed', 'Vortex'))
parser.add_argument('--integrator', default='RK4', choices=('RK4', 'ForwardEuler', 'AB2'))
parser.add_argument('--make_profile', default=0, help='Enable cProfile profiler')
parser.add_argument('--nu', default=0.000625, type=float, help='Viscosity')
parser.add_argument('--dt', default=0.01, type=float, help='Time step size')
parser.add_argument('--T', default=0.1, type=float, help='End time')
parser.add_argument('--M', default=6, type=int, help='Mesh size')
parser.add_argument('--write_result', default=1e8, type=int, help="Write results to HDF5 every...")
parser.add_argument('--write_yz_slice',  default=[0, 1e8], help="Write 2D slice to HDF5 [x index, every]")

dimensions = 3
vars().update(vars(parser.parse_args()))

def update(new):
    assert isinstance(new, dict)
    parser.set_defaults(**new)
    globals().update(vars(parser.parse_args()))
