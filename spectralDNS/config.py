"""Parameters for the spectralDNS solvers

The parameters are kept in dictionary 'params'. The items of this 
dictionary may be accessed as attributes, e.g., 

M = config,params.M  does the same thing as M = config.params['M']

Generic parameters for all solvers::
    precision     (str)              ('double', 'single') 
    optimization  (str)              ('cython', 'numba', None)
    make_profile  (int)              Whether on not to enable profiling
    dt            (float)            Time step for fixed time step integrators
    T             (float)            End time
    nu            (float)            Viscosity
    t             (float)            Time
    tstep         (int)              Time step
    L        (float, float(, float)) Domain size (2 for 2D, 3 for 3D)
    M             (int, int(, int))  Mesh size   (2 for 2D, 3 for 3D)
    write_result  (int)              Store results as HDF5 every (*) time step
    checkpoint    (int)              Save intermediate result every (*)
    dealias       (str)              ('3/2-rule', '2/3-rule', 'None')
    ntol          (int)              Tolerance (number of accurate digits used in tests)
    threads       (int)              Number of threads used for FFTs

Parameters for 3D solvers in triply periodic domain::
    convection       (str)           ('Standard', 'Divergence', 'Skewed', 'Vortex')
    decomposition    (str)           ('slab', 'pencil')
    communication    (str)           ('alltoall', 'sendrecv_replace')
    pencil_alignment (str)           ('X', 'Y') Final alignment direction for spectral data 
    P1               (int)           Pencil decomposition in first direction
    write_yz_slice   (int, int)      Store yz slice at x index (*0) every (*1) time step
    write_xz_slice   (int, int)      Store xz slice at y index (*0) every (*1) time step
    write_xy_slice   (int, int)      Store xy slice at z index (*0) every (*1) time step
    integrator       (str)           ('RK4', 'ForwardEuler', 'AB2', 'BS5_adaptive', 'BS5_fixed')
    TOL              (float)         Accuracy used in BS5_adaptive
        
Parameters for 3D solvers in channel domain::
    convection    (str)              ('Standard', 'Divergence', 'Skewed', 'Vortex')
    dealias_cheb  (bool)             Whether or not to dealias in inhomogeneous direction
    decomposition (str)              ('slab',)
    write_yz_slice   (int, int)      Store yz slice at x index (*0) every (*1) time step
    write_xz_slice   (int, int)      Store xz slice at y index (*0) every (*1) time step
    write_xy_slice   (int, int)      Store xy slice at z index (*0) every (*1) time step
    
Parameters for 2D solvers in doubly periodic domain::
    integrator    (str)              ('RK4', 'ForwardEuler', 'AB2', 'BS5_adaptive', 'BS5_fixed')
    decomposition (str)              ('line')
        
Solver specific parameters triply periodic domain::
    MHD::
        eta           (float)        Model parameter
        
Solver specific parameters doubly periodic domain::
    Bq2D::
        Ri            (float)        Model parameter (Richardson number)
        Pr            (float)        Model parameter (Prandtl number)
        integrator    (str)          ('RK4', 'ForwardEuler', 'AB2', 'BS5_adaptive', 'BS5_fixed')
        
Solver specifi parameters channel domain::
    IPCS, IPCSR::
        velocity_pressure_iters   (int)   Number of inner velocity pressure iterations
        print_divergence_progress (bool)  Print the norm of the pressure correction on inner iterations
        divergence_tol            (float) Tolerance on divergence error for pressure velocity coupling
        
"""
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-08"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

import argparse
from numpy import pi, array, float32, float64
import copy
import collections
import json

class Params(collections.MutableMapping, dict):    
    def __init__(self, *args, **kwargs):
        super(Params, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
    def __getattr__(self, key):
        # Called if key is missing in __getattribute__
        if key == 'dx':
            assert self.has_key('M') and self.has_key('L')
            val = self['L']/2**self['M']
            return val
        
        elif key == 'N':
            mval = self['M']
            val = 2**mval
            return val

        else:
            raise KeyError

    def __getattribute__(self, key):
        if key in ('nu', 'dt', 'Ri', 'Pr', 'eta'):
            float = float32 if self['precision'] == 'single' else float64
            return float(dict.__getattribute__(self, key))
        else:
            return dict.__getattribute__(self, key)

    def __setattr__(self, key, val):
        if key in ('M', 'L'):
            self.__setitem__(key, val)
        else:
            dict.__setattr__(self, key, val)
        
    def __getitem__(self, key):
        return dict.__getitem__(self, key)
    
    def __setitem__(self, key, val):
        if key == 'M':
            val = array([eval(str(f)) for f in val], dtype=int)
            val.flags.writeable = False
            dict.__setitem__(self, key, val)
            
        elif key == 'L':
            val = array([eval(str(f)) for f in val], dtype=float)
            val.flags.writeable = False
            dict.__setitem__(self, key, val)
        
        else:
            dict.__setitem__(self, key, val)
         
    def __delitem__(self, key):
        dict.__delitem__(self, key)
        
    def __iter__(self):
        return dict.__iter__(self)
    
    def __len__(self):
        return dict.__len__(self)
    
    def __contains__(self, x):
        return dict.__contains__(self, x)

fft_plans = collections.defaultdict(lambda : "FFTW_MEASURE", {'dct': "FFTW_EXHAUSTIVE"})

class PlanAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        global fft_plans
        fft_plans.update(json.loads(values))
        setattr(namespace, self.dest, fft_plans)

params = Params()

parser = argparse.ArgumentParser(prog='spectralDNS', add_help=False)

parser.add_argument('--precision', default='double', choices=('single', 'double'))
parser.add_argument('--optimization', default='', choices=('cython', 'weave', 'numba'))
parser.add_argument('--make_profile', default=0, type=int, help='Enable cProfile profiler')
parser.add_argument('--dt', default=0.01, type=float, help='Time step size')
parser.add_argument('--T', default=0.1, type=float, help='End time')
parser.add_argument('--write_result', default=1e8, metavar=('tstep'), type=int, help='Write results to HDF5 every tstep')
parser.add_argument('--checkpoint',  default=1e8, type=int, help='Save intermediate result every...')
parser.add_argument('--nu', default=0.000625, type=float, help='Viscosity')
parser.add_argument('--t', default=0.0, type=float, help='Time')
parser.add_argument('--tstep', default=0, type=int, help='Time step')
parser.add_argument('--dealias', default='2/3-rule', choices=('2/3-rule', '3/2-rule', 'None'), help='Choose dealiasing method')
parser.add_argument('--ntol', default=7, type=int, help='Tolerance - number of accurate digits')
parser.add_argument('--threads', default=1, type=int, help='Number of threads used for FFTs')
parser.add_argument('--planner_effort', action=PlanAction, default=fft_plans, 
                    help="""Planning effort for FFTs. Usage, e.g., --planner_effort '{"dct":"FFTW_EXHAUSTIVE"}' """)

# Arguments for isotropic DNS solver
triplyperiodic = argparse.ArgumentParser(parents=[parser])
triplyperiodic.add_argument('--convection', default='Vortex', choices=('Standard', 'Divergence', 'Skewed', 'Vortex'))
triplyperiodic.add_argument('--communication', default='Alltoallw', choices=('Alltoallw', 'alltoall', 'Sendrecv_replace', 'Swap', 'Nyquist'), help='Choose method for communication. sendrecv_replace is only for slab without padding. Swap and Nyquist are only for pencil decomposition.')
triplyperiodic.add_argument('--L', default=[2*pi, 2*pi, 2*pi], metavar=("Lx", "Ly", "Lz"), nargs=3, help='Physical mesh size')
triplyperiodic.add_argument('--Pencil_alignment', default='Y', choices=('X', 'Y'), help='Alignment of the complex data for pencil decomposition')
triplyperiodic.add_argument('--Pencil_P1', default=2, type=int, help='pencil decomposition in first direction')
triplyperiodic.add_argument('--decomposition', default='slab', choices=('slab', 'pencil'), help="Choose 3D decomposition between slab and pencil.")
triplyperiodic.add_argument('--M', default=[6, 6, 6], metavar=("Mx", "My", "Mz"), nargs=3, help='Mesh size is pow(2, M[i]) in direction i')
triplyperiodic.add_argument('--write_yz_slice',  default=[0, 1e8], nargs=2, type=int, metavar=('i', 'tstep'), 
                            help='Write 2D slice of yz plane with index i in x-direction every tstep. ')
triplyperiodic.add_argument('--write_xz_slice',  default=[0, 1e8], nargs=2, type=int, metavar=('j', 'tstep'), 
                            help='Write 2D slice of xz plane with index j in y-direction every tstep. ')
triplyperiodic.add_argument('--write_xy_slice',  default=[0, 1e8], nargs=2, type=int, metavar=('k', 'tstep'), 
                            help='Write 2D slice of xy plane with index k in z-direction every tstep. ')
triplyperiodic.add_argument('--TOL', type=float, default=1e-6, help='Tolerance for adaptive time integrator')
triplyperiodic.add_argument('--integrator', default='RK4', choices=('RK4', 'ForwardEuler', 'AB2', 'BS5_adaptive', 'BS5_fixed'))

trippelsubparsers = triplyperiodic.add_subparsers(dest='solver')

# Remember! Subparser arguments must be invoked after the positional argument
# E.g, python TG.py --M 6 6 6 NS --integrator RK4

parser_NS = trippelsubparsers.add_parser('NS', help='Regular Navier Stokes solver')
parser_VV = trippelsubparsers.add_parser('VV', help='Velocity-Vorticity formulation')
parser_MHD = trippelsubparsers.add_parser('MHD', help='Magnetohydrodynamics solver')
parser_MHD.add_argument('--eta', default=0.01, type=float, help='MHD parameter')

doublyperiodic = argparse.ArgumentParser(parents=[parser])
doublyperiodic.add_argument('--integrator', default='RK4', choices=('RK4', 'ForwardEuler', 'AB2'))
doublyperiodic.add_argument('--L', default=[2*pi, 2*pi], nargs=2, help='Physical mesh size')
doublyperiodic.add_argument('--decomposition', default='line', choices=('line', ), help="For 2D problems line is the only choice.")
doublyperiodic.add_argument('--M', default=[6, 6], nargs=2, help='Mesh size is pow(2, M[i]) in direction i')

doublesubparsers = doublyperiodic.add_subparsers(dest='solver')

parser_NS2D = doublesubparsers.add_parser('NS2D', help='Regular 2D Navier Stokes solver')
parser_Bq2D = doublesubparsers.add_parser('Bq2D', help='Regular 2D Navier Stokes solver with Boussinesq model.')
parser_Bq2D.add_argument('--Ri', default=0.1, type=float, help='Richardson number')
parser_Bq2D.add_argument('--Pr', default=1.0, type=float, help='Prandtl number')

# Arguments for Shen based solver with one inhomogeneous direction
channel = argparse.ArgumentParser(parents=[parser])
channel.add_argument('--convection', default='Vortex', choices=('Standard', 'Divergence', 'Skew', 'Vortex'))
channel.add_argument('--L', default=[2, 2*pi, 2*pi], nargs=3, help='Physical mesh size')
dealias_cheb_parser = channel.add_mutually_exclusive_group(required=False)
dealias_cheb_parser.add_argument('--no-dealias_cheb', dest='dealias_cheb', action='store_false', help='No dealiasing for inhomogeneous direction (default)')
dealias_cheb_parser.add_argument('--dealias_cheb', dest='dealias_cheb', action='store_true', help='Use dealiasing for inhomogeneous direction. If True, use same rule as for periodic')
channel.set_defaults(dealias_cheb=False)
channel.add_argument('--decomposition', default='slab', choices=('slab', 'pencil'), help="Choose 3D decomposition between slab and pencil.")
channel.add_argument('--communication', default='Alltoallw', choices=('Alltoallw', 'alltoall'), help='Choose method for communication.')
channel.add_argument('--Pencil_alignment', default='X', choices=('X',), help='Alignment of the complex data for pencil decomposition')
channel.add_argument('--M', default=[6, 6, 6], nargs=3, help='Mesh size is pow(2, M[i]) in direction i')
channel.add_argument('--write_yz_slice',  default=[0, 1e8], nargs=2, type=int, metavar=('i', 'tstep'), 
                     help='Write 2D slice of yz plane with index i in x-direction every tstep. ')
channel.add_argument('--write_xz_slice',  default=[0, 1e8], nargs=2, type=int, metavar=('j', 'tstep'), 
                     help='Write 2D slice of xz plane with index j in y-direction every tstep. ')
channel.add_argument('--write_xy_slice',  default=[0, 1e8], nargs=2, type=int, metavar=('k', 'tstep'), 
                     help='Write 2D slice of xy plane with index k in z-direction every tstep. ')
channel.add_argument('--Dquad', default='GC', choices=('GC', 'GL'), help="Choose quadrature scheme for Dirichlet space. GC = Chebyshev-Gauss (x_k=cos((2k+1)/(2N+2)*pi)) and GL = Gauss-Lobatto (x_k=cos(k*pi/N))")
channel.add_argument('--Bquad', default='GC', choices=('GC', 'GL'), help="Choose quadrature scheme for Biharmonic space. GC = Chebyshev-Gauss (x_k=cos((2k+1)/(2N+2)*pi)) and GL = Gauss-Lobatto (x_k=cos(k*pi/N))")
channel.add_argument('--Nquad', default='GC', choices=('GC', 'GL'), help="Choose quadrature scheme for Neumann space. GC = Chebyshev-Gauss (x_k=cos((2k+1)/(2N+2)*pi)) and GL = Gauss-Lobatto (x_k=cos(k*pi/N))")
channelsubparsers = channel.add_subparsers(dest='solver')

KMM = channelsubparsers.add_parser('KMM', help='Kim Moin Moser channel solver with Crank-Nicolson and Adams-Bashforth discretization.')
KMMRK3 = channelsubparsers.add_parser('KMMRK3', help='Kim Moin Moser channel solver with third order semi-implicit Runge-Kutta discretization.')
IPCS = channelsubparsers.add_parser('IPCS', help='Incremental pressure correction with Crank-Nicolson and Adams-Bashforth discretization.')
IPCS.add_argument('--velocity_pressure_iters', default=1, type=int, help='Number of inner velocity pressure iterations for IPCS')
print_div_parser = IPCS.add_mutually_exclusive_group(required=False)
print_div_parser.add_argument('--no-print_divergence_progress', dest='print_divergence_progress', action='store_false')
print_div_parser.add_argument('--print_divergence_progress', dest='print_divergence_progress', action='store_true', help='Print the norm of the pressure correction on inner iterations for IPCS')
IPCS.set_defaults(print_divergence_progress=False)
IPCS.add_argument('--divergence_tol', default=1e-7, type=float, help='Tolerance on divergence error for pressure velocity coupling for IPCS')
IPCSR = channelsubparsers.add_parser('IPCSR', help='Incremental pressure correction with Crank-Nicolson and Adams-Bashforth discretization.')
IPCSR.add_argument('--velocity_pressure_iters', default=1, type=int, help='Number of inner velocity pressure iterations for IPCS')
print_div_parser = IPCSR.add_mutually_exclusive_group(required=False)
print_div_parser.add_argument('--no-print_divergence_progress', dest='print_divergence_progress', action='store_false')
print_div_parser.add_argument('--print_divergence_progress', dest='print_divergence_progress', action='store_true', help='Print the norm of the pressure correction on inner iterations for IPCSR')
IPCSR.set_defaults(print_divergence_progress=False)
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
    global fft_plans
    assert isinstance(new, dict)
    if 'planner_effort' in new:
        fft_plans.update(new['planner_effort'])
        new['planner_effort'] = fft_plans
        
    exec mesh + ".set_defaults(**new)"

