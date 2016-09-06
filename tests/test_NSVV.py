import pytest
from spectralDNS import config, get_solver, solve
from TG import initialize, regression_test, pi
from mpi4py import MPI

comm = MPI.COMM_WORLD

if comm.Get_size() >= 4:
    params = ('NS_uniform_slab', 'VV_uniform_slab',
              'NS_nonuniform_slab', 'VV_nonuniform_slab',
              'NS_uniform_pencil', 'VV_uniform_pencil',
              'NS_nonuniform_pencil', 'VV_nonuniform_pencil')
else:
    params = ('NS_uniform', 'VV_uniform',
              'NS_nonuniform', 'VV_nonuniform')

@pytest.fixture(params=params)
def sol(request):
    """Check for uniform and non-uniform cube"""
    pars = request.param.split('_')
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
    initialize(solver, **context)
    solve(solver, context)

    config.params.make_profile = 1
    config.params.dealias = '3/2-rule'
    initialize(solver, **context)
    solve(solver, context)
    
    config.params.dealias = '2/3-rule'
    config.params.optimization = 'cython'
    initialize(solver, **context)
    solve(solver, context)
    
    config.params.write_result = 1
    config.params.checkpoint = 1
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

    solver = get_solver(regression_test=regression_test,
                        parse_args=sol)
    context = solver.get_context()
    for integrator in ('RK4', 'ForwardEuler', 'AB2', 'BS5_adaptive', 'BS5_fixed'):
        if integrator in ('ForwardEuler', 'AB2'):
            config.params.ntol = 4
        else:
            config.params.ntol = 7
        config.params.integrator = integrator
        initialize(solver, **context)
        solve(solver, context)

if __name__ == '__main__':
    test_solvers(['NS'])
    test_integrators(['NS'])
