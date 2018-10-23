import pytest
from six.moves import reload_module
from mpi4py import MPI
from spectralDNS import config, get_solver, solve
from TG import initialize, regression_test

comm = MPI.COMM_WORLD

if comm.Get_size() >= 4:
    params = ('NS/uniform/slab', 'VV/uniform/slab',
              'NS/nonuniform/slab', 'VV/nonuniform/slab',
              'NS/uniform/pencil', 'VV/uniform/pencil',
              'NS/nonuniform/pencil', 'VV/nonuniform/pencil',
              'NS_shenfun/uniform/slab',
              'VV_shenfun/uniform/slab',
              'NS_shenfun/nonuniform/slab',
              'VV_shenfun/nonuniform/slab',
              'NS_shenfun/uniform/pencil',
              'VV_shenfun/uniform/pencil',
              'NS_shenfun/nonuniform/pencil',
              'VV_shenfun/nonuniform/pencil')
else:
    params = ('NS/uniform', 'VV/uniform',
              'NS/nonuniform', 'VV/nonuniform',
              'NS_shenfun/uniform',
              'VV_shenfun/uniform',
              'NS_shenfun/nonuniform',
              'VV_shenfun/nonuniform')

@pytest.fixture(params=params)
def sol(request):
    """Check for uniform and non-uniform cube"""
    pars = request.param.split('/')
    solver, mesh = pars[:2]
    mpi = 'slab'
    if len(pars) == 3:
        mpi = pars[2]
    _args = ['--decomposition', mpi]
    if mesh == 'uniform':
        _args += ['--M', '4', '4', '4', '--L', '2*pi', '2*pi', '2*pi']
    else:
        _args += ['--M', '6', '5', '4', '--L', '6*pi', '4*pi', '2*pi']
    _args += [solver]

    return _args


def test_solvers(sol):
    config.update(
        {
            'nu': 0.000625,             # Viscosity
            'dt': 0.01,                 # Time step
            'T': 0.1,                    # End time
            'convection': 'Vortex'
        }
    )

    solver = get_solver(regression_test=regression_test,
                        parse_args=sol)
    context = solver.get_context()
    initialize(solver, context)
    solve(solver, context)

    config.params.make_profile = 1
    config.params.dealias = '3/2-rule'
    initialize(solver, context)
    solve(solver, context)

    config.params.dealias = '2/3-rule'
    for opt in ('cython', 'numba', 'pythran'):
        config.params.optimization = opt
        reload_module(solver)  # To load optimized methods
        initialize(solver, context)
        solve(solver, context)

    config.params.write_result = 2
    config.params.checkpoint = 2
    config.params.dt = 0.01
    config.params.t = 0.0
    config.params.tstep = 0
    config.params.T = 0.04
    solver.regression_test = lambda c: None
    solve(solver, context)

def test_integrators(sol):
    config.update(
        {
            'nu': 0.000625,             # Viscosity
            'dt': 0.01,                 # Time step
            'T': 0.1,                    # End time
            'convection': 'Vortex'
        }
    )

    solver = get_solver(regression_test=regression_test, parse_args=sol)
    context = solver.get_context()
    for integrator in ('RK4', 'ForwardEuler', 'AB2', 'BS5_adaptive', 'BS5_fixed'):
        if integrator in ('ForwardEuler', 'AB2'):
            config.params.ntol = 4
        else:
            config.params.ntol = 7
        config.params.integrator = integrator
        initialize(solver, context)
        solve(solver, context)

if __name__ == '__main__':
    test_solvers(['NS'])
    #test_integrators(['NS'])
