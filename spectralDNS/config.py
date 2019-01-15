"""Parameters for the spectralDNS solvers

The parameters are kept in dictionary 'params'. The values of this
dictionary may be accessed as attributes, e.g.,

M = config.params.M  does the same thing as M = config.params['M']

Generic parameters for all solvers::
    precision        (str)           ('double', 'single')
    optimization     (str)           ('cython', 'numba', None)
    make_profile     (int)           Whether on not to enable profiling
    dt               (float)         Time step for fixed time step integrators
    T                (float)         End time
    nu               (float)         Viscosity
    t                (float)         Time
    tstep            (int)           Time step
    L        (float, float(, float)) Domain size (2 for 2D, 3 for 3D)
    M             (int, int(, int))  Mesh size   (2 for 2D, 3 for 3D)
    write_result     (int)           Store results as HDF5 every (*) time step
    checkpoint       (int)           Save intermediate result every (*)
    dealias          (str)           ('3/2-rule', '2/3-rule', 'None')
    decomposition    (str)           ('slab', 'pencil')
    ntol             (int)           Tolerance (number of accurate digits used in tests)
    threads          (int)           Number of threads used for FFTs
    h5filename       (str)           Filename for storing HDF5 results
    verbose          (bool)          Print some timings in the end
    convection       (str)           ('Standard', 'Divergence', 'Skewed', 'Vortex')

Parameters for 3D explicit solvers::
    integrator       (str)           ('RK4', 'ForwardEuler', 'AB2', 'BS5_adaptive', 'BS5_fixed')
    TOL              (float)         Accuracy used in BS5_adaptive

Solver specific parameters triply periodic domain::
    MHD::
        eta          (float)         Model parameter

Solver specific parameters doubly periodic domain::
    Bq2D::
        Ri           (float)         Model parameter (Richardson number)
        Pr           (float)         Model parameter (Prandtl number)

"""
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-08"
__copyright__ = "Copyright (C) 2015-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

import argparse
import collections
import json
from numpy import pi, array, float32, float64

#pylint: disable=global-statement,redefined-outer-name,exec-used

class AttributeDict(collections.MutableMapping, dict):
    """Dictionary class

    The values of this dictionary may be accessed as attributes:

        p = Params({'M': 2})
        M = p.M
        N = p['M']
        assert M is N

    """
    def __init__(self, *args, **kwargs):
        super(AttributeDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, key):
        return dict.__getattribute__(self, key)

    def __setattr__(self, key, val):
        dict.__setattr__(self, key, val)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def __delitem__(self, key):
        dict.__delitem__(self, key)

    def __iter__(self):
        return dict.__iter__(self)

    def __len__(self):
        return dict.__len__(self)

    def __contains__(self, x):
        return dict.__contains__(self, x)


class Params(AttributeDict):
    """Class for collection of parameters

    The values of this dictionary may be accessed as attributes:

        p = Params({'M': 2})
        M = p.M
        N = p['M']
        assert M is N

    """
    def __init__(self, *args, **kwargs):
        AttributeDict.__init__(self, *args, **kwargs)

    def __getattr__(self, key):
        # Called if key is missing in __getattribute__
        if key == 'dx':
            return self.L / self.N

        elif key == 'N':
            assert 'M' in self
            mval = self.M
            return 2**mval

        else:
            raise KeyError

    def __getattribute__(self, key):
        if key in ('nu', 'dt', 'Ri', 'Pr', 'eta'):
            fl = float32 if self['precision'] == 'single' else float64
            return fl(dict.__getattribute__(self, key))
        return dict.__getattribute__(self, key)

    def __setattr__(self, key, val):
        if key in ('N', 'M', 'L'):
            self.__setitem__(key, val)
        else:
            dict.__setattr__(self, key, val)

    def __setitem__(self, key, val):
        if key in ('M', 'N'):
            val = array([int(str(f)) for f in val], dtype=int)
            val.flags.writeable = False
            dict.__setitem__(self, key, val)

        elif key == 'L':
            val = array([eval(str(f), {"__builtins__": None}, {'pi': pi}) for f in val],
                        dtype=float)
            val.flags.writeable = False
            dict.__setitem__(self, key, val)

        else:
            dict.__setitem__(self, key, val)

fft_plans = collections.defaultdict(lambda: "FFTW_MEASURE",
                                    {'dct': "FFTW_MEASURE"})

class PlanAction(argparse.Action):
    """Action for planning FFT"""
    def __call__(self, parser, namespace, values, option_string=None):
        global fft_plans
        fft_plans.update(json.loads(values))
        setattr(namespace, self.dest, fft_plans)

# Create an instance of the Params class to hold all parameters for the solvers
params = Params()

# Create the main parser
parser = argparse.ArgumentParser(prog='spectralDNS', add_help=False)

# Arguments used by all solvers
parser.add_argument('--precision', default='double',
                    choices=('single', 'double'))
parser.add_argument('--optimization', default='',
                    choices=('cython', 'weave', 'numba', 'pythran'),
                    help='Choose implementation method for optimization')
parser.add_argument('--make_profile', default=0, type=int,
                    help='Enable cProfile profiler')
parser.add_argument('--dt', default=0.01, type=float,
                    help='Time step size')
parser.add_argument('--T', default=0.1, type=float,
                    help='End time')
parser.add_argument('--write_result', default=1e8, metavar=('tstep'), type=int,
                    help='Write results to HDF5 every tstep')
parser.add_argument('--checkpoint', default=1e8, type=int,
                    help='Save intermediate result every...')
parser.add_argument('--nu', default=0.000625, type=float,
                    help='Viscosity')
parser.add_argument('--t', default=0.0, type=float,
                    help='Time')
parser.add_argument('--tstep', default=0, type=int,
                    help='Time step')
parser.add_argument('--filemode', default='w',
                    choices=('w', 'r', 'a'),
                    help='Choose mode for opening HDF5 files')
parser.add_argument('--dealias', default='2/3-rule',
                    choices=('2/3-rule', '3/2-rule', 'None'),
                    help='Choose dealiasing method')
parser.add_argument('--decomposition', default='slab', choices=('slab', 'pencil'),
                    help="Choose MPI decomposition between slab and pencil.")
parser.add_argument('--ntol', default=7, type=int,
                    help='Tolerance - number of accurate digits')
parser.add_argument('--threads', default=1, type=int,
                    help='Number of threads used for FFTs')
parser.add_argument('--planner_effort', action=PlanAction, default=fft_plans,
                    help="""Planning effort for FFTs. Usage, e.g., --planner_effort '{"dct":"FFTW_EXHAUSTIVE"}' """)
parser.add_argument('--h5filename', default='results', type=str,
                    help='Filename of HDF5 datafile used to store intermediate checkpoint data or timeseries results')
parser.add_argument('--verbose', dest='verbose', action='store_true', help='Print timings in the end')
parser.add_argument('--no-verbose', dest='verbose', action='store_false', help='Do not print timings in the end')
parser.set_defaults(verbose=True)

# Arguments for 3D isotropic solvers
triplyperiodic = argparse.ArgumentParser(parents=[parser])

triplyperiodic.add_argument('--convection', default='Vortex',
                            choices=('Standard', 'Divergence', 'Skewed', 'Vortex'),
                            help='Choose method for computing the nonlinear convective term')
triplyperiodic.add_argument('--L', default=[2*pi, 2*pi, 2*pi], metavar=("Lx", "Ly", "Lz"), nargs=3,
                            help='Physical mesh size')

triplyperiodic.add_argument('--M', default=[6, 6, 6], metavar=("Mx", "My", "Mz"), nargs=3,
                            help='Mesh size is pow(2, M[i]) in direction i. Used if N is missing.')
triplyperiodic.add_argument('--TOL', type=float, default=1e-6,
                            help='Tolerance for adaptive time integrator')
triplyperiodic.add_argument('--integrator', default='RK4',
                            choices=('RK4', 'ForwardEuler', 'AB2', 'BS5_adaptive', 'BS5_fixed'),
                            help='Integrator for triply periodic domain')

trippelsubparsers = triplyperiodic.add_subparsers(dest='solver')

# Remember! Subparser arguments must be invoked after the positional argument
# E.g, python TG.py --M 6 6 6 NS --integrator RK4

parser_NS = trippelsubparsers.add_parser('NS', help='Regular Navier Stokes solver')
parser_VV = trippelsubparsers.add_parser('VV', help='Velocity-Vorticity formulation')
parser_MHD = trippelsubparsers.add_parser('MHD', help='Magnetohydrodynamics solver')
parser_MHD.add_argument('--eta', default=0.01, type=float, help='MHD parameter')
parser_Bq = trippelsubparsers.add_parser('Bq', help='Navier Stokes solver with Boussinesq model')
parser_Bq.add_argument('--Ri', default=0.1, type=float, help='Richardson number')
parser_Bq.add_argument('--Pr', default=1.0, type=float, help='Prandtl number')

# Arguments for 2D periodic solvers
doublyperiodic = argparse.ArgumentParser(parents=[parser])
doublyperiodic.add_argument('--integrator', default='RK4',
                            choices=('RK4', 'ForwardEuler', 'AB2', 'BS5_fixed', 'BS5_adaptive'),
                            help='Integrator for doubly periodic domain')
doublyperiodic.add_argument('--L', default=[2*pi, 2*pi], nargs=2, metavar=('Lx', 'Ly'),
                            help='Physical mesh size')
doublyperiodic.add_argument('--convection', default='Vortex',
                            choices=('Vortex'),
                            help='Choose method for computing the nonlinear convective term')
doublyperiodic.add_argument('--TOL', type=float, default=1e-6,
                            help='Tolerance for adaptive time integrator')
doublyperiodic.add_argument('--M', default=[6, 6], nargs=2, metavar=('Mx', 'My'),
                            help='Mesh size is pow(2, M[i]) in direction i. Used if N is missing.')

doublesubparsers = doublyperiodic.add_subparsers(dest='solver')

parser_NS2D = doublesubparsers.add_parser('NS2D', help='Regular 2D Navier Stokes solver')
parser_Bq2D = doublesubparsers.add_parser('Bq2D', help='Regular 2D Navier Stokes solver with Boussinesq model.')
parser_Bq2D.add_argument('--Ri', default=0.1, type=float, help='Richardson number')
parser_Bq2D.add_argument('--Pr', default=1.0, type=float, help='Prandtl number')

# Arguments for channel solvers with one inhomogeneous direction
channel = argparse.ArgumentParser(parents=[parser])
channel.add_argument('--convection', default='Vortex',
                     choices=('Standard', 'Divergence', 'Skew', 'Vortex'),
                     help='Choose method for computing the nonlinear convective term')
channel.add_argument('--L', default=[2, 2*pi, 2*pi], nargs=3, metavar=('Lx', 'Ly', 'Lz'),
                     help='Physical mesh size')
channel.add_argument('--M', default=[6, 6, 6], nargs=3, metavar=('Mx', 'My', 'Mz'),
                     help='Mesh size is pow(2, M[i]) in direction i. Used if N is missing.')
channel.add_argument('--Dquad', default='GC', choices=('GC', 'GL'),
                     help="Choose quadrature scheme for Dirichlet space. GC = Chebyshev-Gauss (x_k=cos((2k+1)/(2N+2)*pi)) and GL = Gauss-Lobatto (x_k=cos(k*pi/N))")
channel.add_argument('--Bquad', default='GC', choices=('GC', 'GL'),
                     help="Choose quadrature scheme for Biharmonic space. GC = Chebyshev-Gauss (x_k=cos((2k+1)/(2N+2)*pi)) and GL = Gauss-Lobatto (x_k=cos(k*pi/N))")
channel.add_argument('--Nquad', default='GC', choices=('GC', 'GL'),
                     help="Choose quadrature scheme for Neumann space. GC = Chebyshev-Gauss (x_k=cos((2k+1)/(2N+2)*pi)) and GL = Gauss-Lobatto (x_k=cos(k*pi/N))")
channelsubparsers = channel.add_subparsers(dest='solver')

KMM = channelsubparsers.add_parser('KMM', help='Kim Moin Moser channel solver with Crank-Nicolson and Adams-Bashforth discretization.')
KMM.add_argument('--integrator', default='implicit', choices=('implicit',), help='Regular Crank-Nicolson/Adams-Bashforth integrator for channel solver')

KMMr = channelsubparsers.add_parser('KMMr', help='Kim Moin Moser channel solver with Crank-Nicolson and Adams-Bashforth discretization. Inhomogeneous space in z-direction.')
KMMr.add_argument('--integrator', default='implicit', choices=('implicit',), help='Regular Crank-Nicolson/Adams-Bashforth integrator for channel solver')

KMMRK3 = channelsubparsers.add_parser('KMMRK3', help='Kim Moin Moser channel solver with third order semi-implicit Runge-Kutta discretization.')
KMMRK3.add_argument('--integrator', default='implicitRK3', choices=('implicitRK3',), help='RK3 integrator for channel solver')

KMM_RB = channelsubparsers.add_parser('KMM_RB', help='Rayleigh-Benard channel solver using KMM')
KMM_RB.add_argument('--integrator', default='implicit', choices=('implicit',), help='Regular Crank-Nicolson/Adams-Bashforth integrator for channel solver')

KMMRK3_RB = channelsubparsers.add_parser('KMMRK3_RB', help='Rayleigh-Benard channel solver using KMMRK3.')
KMMRK3_RB.add_argument('--integrator', default='implicitRK3', choices=('implicitRK3',), help='RK3 integrator for channel solver')


def update(new, mesh="triplyperiodic"):
    """Update spectralDNS parameters"""
    global fft_plans
    assert isinstance(new, dict)
    if 'planner_effort' in new:
        fft_plans.update(new['planner_effort'])
        new['planner_effort'] = fft_plans
    globals()[mesh].set_defaults(**new)
