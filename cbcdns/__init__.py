__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-09"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from utilities.MPI_knee import mpi_import
with mpi_import():
    from cbcdns import config

    if config.solver == 'NS':
        import solvers.spectralDNS as solver
        
    elif config.solver == 'VV':
        import solvers.spectralDNSVV as solver
        
    elif config.solver == 'NS2D':
        import solvers.spectralDNS2D as solver
        
    elif config.solver == 'MHD':
        import solvers.spectralMHD3D as solver
        
