"""
Global run-time configuration that may be overloaded on the commandline
"""
decomposition = 'slab'        # 'slab' or 'pencil'
precision = 'double'          # single or double precision
optimization = None           # cython, weave, numba, numexpr
problem = 'TaylorGreen'       # Name of problem file
dimensions = 3                # 2 or 3
communication = 'alltoall'    # 'alltoall' or 'sendrecv_replace' (only for slab)
convection = 'Vortex'         # 'Standard', 'Divergence', 'Skewed', 'Vortex'
integrator = 'RK4'            # Integrator ('RK4', 'ForwardEuler', 'AB2')
make_profile = 0              # Enable cProfile profiler
nu = 0.000625                 # Viscosity
dt = 0.01                     # Time step
T = 0.1                       # End time
M = 6                         # Mesh size
write_result = 2
write_yz_slice = [0, 1e8]

def update(new):
    assert isinstance(new, dict)
    d = globals()
    for key, val in new.iteritems():
        if key in d:
            d[key] = val
        else:
            pass

    assert d['optimization'] in (None, 'weave', 'cython', 'numexpr', 'numba')
    assert d['precision'] in ('single', 'double')
    assert d['decomposition'] in ('slab', 'pencil')
    assert d['dimensions'] in (2, 3)
    assert d['convection'] in ('Standard', 'Divergence', 'Skewed', 'Vortex')
    assert d['integrator'] in ('RK4', 'ForwardEuler', 'AB2')
    assert d['communication'] in ('alltoall', 'sendrecv_replace')
