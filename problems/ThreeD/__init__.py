parameters = {
    'M': 5,                     # Mesh size
    'P1': 1,                    # Mesh decomposition in first direction (pencil P1*P2=num_processes)
    'write_result': 1e8,        # Write to HDF5 every..
    'write_yz_slice': [0, 1e8], # Write slice 0 (or higher) in y-z plance every..
    'nu': 0.000625,             # Viscosity
    'dt': 0.01,                 # Time step
    'T': 0.1,                   # End time
}

def set_source(Source, **kwargs):
    Source[:] = 0
    return Source

def initialize(U, **kwargs):
    return U

def update(**kwargs):
    pass

def finalize(**kwargs):
    pass

