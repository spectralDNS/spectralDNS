parameters = {
    'problem': 'TaylorGreen',   # Decide the problem to solve
    'decomposition': 'slab',    # 'slab' or 'pencil'
    'communication': 'alltoall',# 'alltoall' or 'sendrecv_replace' (only for slab)
    'convection': 'Vortex',     # 'Standard', 'Divergence', 'Skewed', 'Vortex'
    'make_profile': 0,          # Enable cProfile profiler
    'mem_profile': False,       # Check memory use
    'M': 5,                     # Mesh size
    'P1': 1,                    # Mesh decomposition in first direction (pencil P1*P2=num_processes)
    'integrator': 'RK4',        # Integrator ('RK4', 'ForwardEuler', 'AB2')
    'write_result': 1e8,        # Write to HDF5 every..
    'write_yz_slice': [0, 1e8], # Write slice 0 (or higher) in y-z plance every..
    'compute_energy': 2,        # Compute solution energy every..
    'nu': 0.000625,             # Viscosity
    'dt': 0.01,                 # Time step
    'T': 0.1,                   # End time
    'precision': "double",      # single or double precision
    'optimization': None        # Choose optimization None, weave, cython
}

def check_parameters(par):
    assert par['convection'] in ('Standard', 'Divergence', 'Skewed', 'Vortex')
    assert par['integrator'] in ('RK4', 'ForwardEuler', 'AB2')
    assert par['precision'] in ('single', 'double')
    assert par['communication'] in ('alltoall', 'sendrecv_replace')
    assert par['optimization'] in (None, "weave", "cython")
