__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *

hdf5file = HDF5Writer(comm, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, config.solver+".h5")

def standardConvection(c):
    """c_i = u_j du_i/dx_j"""
    for i in range(3):
        for j in range(3):
            U_tmp[j] = ifftn_mpi(1j*K[j]*U_hat[i], U_tmp[j])
        c[i] = fftn_mpi(sum(U*U_tmp, 0), c[i])
    return c

def divergenceConvection(c, add=False):
    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    for i in range(3):
        F_tmp[i] = fftn_mpi(U[0]*U[i], F_tmp[i])
    c[0] += 1j*sum(K*F_tmp, 0)
    c[1] += 1j*K[0]*F_tmp[1]
    c[2] += 1j*K[0]*F_tmp[2]
    F_tmp[0] = fftn_mpi(U[1]*U[1], F_tmp[0])
    F_tmp[1] = fftn_mpi(U[1]*U[2], F_tmp[1])
    F_tmp[2] = fftn_mpi(U[2]*U[2], F_tmp[2])
    c[1] += (1j*K[1]*F_tmp[0] + 1j*K[2]*F_tmp[1])
    c[2] += (1j*K[1]*F_tmp[1] + 1j*K[2]*F_tmp[2])
    return c

#@profile
def Cross(a, b, c):
    """c_k = F_k(a x b)"""
    U_tmp[:] = cross1(U_tmp, a, b)
    c[0] = fftn_mpi(U_tmp[0], c[0])
    c[1] = fftn_mpi(U_tmp[1], c[1])
    c[2] = fftn_mpi(U_tmp[2], c[2])
    return c

#@profile
def Curl(a, c):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    F_tmp[:] = cross2(F_tmp, K, a)
    c[0] = ifftn_mpi(F_tmp[0], c[0])
    c[1] = ifftn_mpi(F_tmp[1], c[1])
    c[2] = ifftn_mpi(F_tmp[2], c[2])    
    return c

def getConvection(convection):
    """Return function used to compute convection"""
    if convection == "Standard":
        
        def Conv(dU):
            dU = standardConvection(dU)
            dU[:] *= -1 
            return dU
        
    elif convection == "Divergence":
        
        def Conv(dU):
            dU = divergenceConvection(dU, False)
            dU[:] *= -1
            return dU
        
    elif convection == "Skewed":
        
        def Conv(dU):
            dU = standardConvection(dU)
            dU = divergenceConvection(dU, True)        
            dU *= -0.5
            return dU
        
    elif convection == "Vortex":
        
        def Conv(dU):
            curl[:] = Curl(U_hat, curl)
            dU = Cross(U, curl, dU)
            return dU
        
    return Conv           

conv = getConvection(config.convection)

@optimizer
def add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu):
    """Add contributions from pressure and diffusion to the rhs"""
    
    # Compute pressure (To get actual pressure multiply by 1j)
    P_hat = sum(dU*K_over_K2, 0, out=P_hat)
        
    # Subtract pressure gradient
    dU -= P_hat*K
    
    # Subtract contribution from diffusion
    dU -= nu*K2*U_hat
    
    return dU

#@profile
def ComputeRHS(dU, rk):
    """Compute and return entire rhs contribution"""
    
    if rk > 0: # For rk=0 the correct values are already in U
        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])
                        
    dU = conv(dU)

    #dU = dealias_rhs(dU, dealias)
    
    dU = add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
        
    return dU

def regression_test(t, tstep, **kw):
    pass

# Set up function to perform temporal integration (using config.integrator parameter)
integrate = getintegrator(**vars())

def solve():
    timer = Timer()
    t = 0.0
    tstep = 0
    while t < config.T-1e-8:
        t += dt 
        tstep += 1
        
        U_hat[:] = integrate(t, tstep, dt)

        for i in range(3):
            U[i] = ifftn_mpi(U_hat[i], U[i])
                 
        update(t, tstep, **globals())
        
        timer()
        
        if tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()

    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**globals())
        
    regression_test(t, tstep, **globals())
        
    hdf5file.close()
