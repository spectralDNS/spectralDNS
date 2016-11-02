import pytest
from spectralDNS import config, get_solver, solve
from OrrSommerfeld import initialize, regression_test, set_Source, pi

@pytest.fixture(params=('KMMRK3', 'KMM', 'IPCS', 'IPCSR'))
def sol(request):
    """Check for uniform and non-uniform cube"""
    return request.param

def test_channel(sol):
    config.update(
        {
            'Re': 8000.,
            'nu': 1./8000.,             # Viscosity
            'dt': 0.001,                 # Time step
            'T': 0.01,                   # End time
            'L': [2, 2*pi, 4*pi/3.],
            'M': [7, 5, 2]
        },  "channel"
    )
    
    solver = get_solver(regression_test=regression_test,
                        mesh="channel",
                        parse_args=[sol])
    print "Hei"
    context = solver.get_context()
    print "Hei1"
    initialize(solver, context)
    print "Hei2"
    set_Source(**context)
    print "Hei3"
    solve(solver, context)
    print "Hei4"

    config.params.dealias = '3/2-rule'
    config.params.optimization = 'cython'
    print "Hei5"
    initialize(solver, context)
    print "Hei6"
    solve(solver, context)
    print "Hei7"

    config.params.dealias_cheb = True
    config.params.checkpoint = 5
    config.params.write_result = 2
    print "Hei8"
    initialize(solver, context)
    print "Hei9"    
    solve(solver, context)
    print "Heiend"

if __name__=='__main__':
    test_channel('KMMRK3')
    test_channel('KMM')
