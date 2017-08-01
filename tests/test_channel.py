import pytest
from spectralDNS import config, get_solver, solve
from six.moves import reload_module
from OrrSommerfeld import initialize, regression_test, set_Source, pi

@pytest.fixture(params=('KMMRK3', 'KMM', 'IPCS', 'IPCSR', 'KMMRK3_mpifft4py', 'KMM_mpifft4py'))
def sol(request):
    """Check for uniform and non-uniform cube"""
    return request.param

def test_channel(sol):
    if sol in ('IPCS', 'IPCSR'):
        pytest.skip(sol+' not currently working')
    config.update(
        {
            'Re': 8000.,
            'nu': 1./8000.,             # Viscosity
            'dt': 0.001,                 # Time step
            'T': 0.01,                   # End time
            'L': [2, 2*pi, 4*pi/3.],
            'M': [7, 5, 2],
            'eps': 1e-7
        },  "channel"
    )

    solver = get_solver(regression_test=regression_test,
                        mesh="channel",
                        parse_args=[sol])
    context = solver.get_context()
    initialize(solver, context)
    set_Source(**context)
    solve(solver, context)

    config.params.dealias = '3/2-rule'
    config.params.optimization = 'cython'
    reload_module(solver) # Need to reload to enable optimization
    initialize(solver, context)
    solve(solver, context)

    config.params.dealias_cheb = True
    config.params.checkpoint = 5
    config.params.write_result = 2
    initialize(solver, context)
    solve(solver, context)

if __name__=='__main__':
    test_channel('IPCS')
