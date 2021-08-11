import pytest
import importlib
from mpi4py import MPI
from spectralDNS import config, get_solver, solve
from TGMHD import initialize, regression_test, pi

comm = MPI.COMM_WORLD

if comm.Get_size() >= 4:
    params = ('uniform_slab', 'nonuniform_slab',
              'uniform_pencil', 'nonuniform_pencil')
else:
    params = ('uniform', 'nonuniform')

@pytest.fixture(params=params)
def sol(request):
    """Check for uniform and non-uniform cube"""
    pars = request.param.split('_')
    mesh = pars[0]
    mpi = 'slab'
    if len(pars) == 2:
        mpi = pars[1]
    _args = ['--decomposition', mpi]
    if mesh == 'uniform':
        _args += ['--M', '4', '4', '4', '--L', '2*pi', '2*pi', '2*pi']
    else:
        _args += ['--M', '6', '5', '4', '--L', '6*pi', '4*pi', '2*pi']
    _args += ['MHD']

    return _args

def test_MHD(sol):
    config.update(
        {
            'nu': 0.000625,             # Viscosity
            'dt': 0.01,                 # Time step
            'T': 0.1,                   # End time
            'eta': 0.01,
            'L': [2*pi, 4*pi, 6*pi],
            'M': [4, 5, 6],
            'convection': 'Divergence'
        }
    )

    solver = get_solver(regression_test=regression_test,
                        parse_args=sol)
    context = solver.get_context()
    initialize(**context)
    solve(solver, context)

    config.params.dealias = '3/2-rule'
    initialize(**context)
    solve(solver, context)

    config.params.dealias = '2/3-rule'
    config.params.optimization = 'cython'
    importlib.reload(solver)
    initialize(**context)
    solve(solver, context)

    config.params.write_result = 1
    config.params.checkpoint = 1
    config.dt = 0.01
    config.params.t = 0.0
    config.params.tstep = 0
    config.T = 0.04
    solver.regression_test = lambda c: None
    solve(solver, context)
