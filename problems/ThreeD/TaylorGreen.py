__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-07"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..ThreeD import *

parameters.update(
    {
      'nu': 0.000625,             # Viscosity
      'dt': 0.01,                 # Time step
      'T': 0.1,                   # End time
      'plot_dkdt': False,
      'compute_energy': 2         # Compute solution energy every..        
    }
)

try:
    from pylab import figure, plot, show
except:
    pass
        
def initialize(X, U, sin, cos, **soak):
    U[0] = sin(X[0])*cos(X[1])*cos(X[2])
    U[1] =-cos(X[0])*sin(X[1])*cos(X[2])
    U[2] = 0 
    return U

k = []
w = []
def update(comm, rank, tstep, write_result, write_yz_slice, P, P_hat, U, curl, 
           float64, dx, L, sum, hdf5file, ifftn_mpi, compute_energy, t, **soak):
    global k, w
    if tstep % write_result == 0 or tstep % write_yz_slice[1] == 0:
        P = ifftn_mpi(P_hat*1j, P)
        hdf5file.write(U, P, tstep)

    if tstep % compute_energy == 0:
        kk = comm.reduce(sum(U.astype(float64)*U.astype(float64))*dx*dx*dx/L**3/2) # Compute energy with double precision
        ww = comm.reduce(sum(curl.astype(float64)*curl.astype(float64))*dx*dx*dx/L**3/2)
        if rank == 0:
            k.append(kk)
            w.append(ww)
            print t, float(kk), float(ww)

def finalize(rank, array, dt, plot_dkdt, **soak):
    global k

    if rank == 0 and plot_dkdt:
        figure()
        k = array(k)
        dkdt = (k[1:]-k[:-1])/dt
        plot(-dkdt)
        show()
