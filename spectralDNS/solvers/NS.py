__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *

vars().update(setupDNS(**vars()))

hdf5file = HDF5Writer(FFT, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, config.solver+".h5")

def standardConvection(c, U_dealiased, dealias=None):
    """c_i = u_j du_i/dx_j"""
    gradUi = work[(U_dealiased, 2)]
    for i in range(3):
        for j in range(3):
            gradUi[j] = FFT.ifftn(1j*K[j]*U_hat[i], gradUi[j], dealias)
        c[i] = FFT.fftn(sum(U_dealiased*gradUi, 0), c[i], dealias)
    return c

def divergenceConvection(c, U_dealiased, dealias=None, add=False):
    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    UUi_hat = work[(c, 0)]
    for i in range(3):
        UUi_hat[i] = FFT.fftn(U_dealiased[0]*U_dealiased[i], UUi_hat[i], dealias)
    c[0] += 1j*sum(K*UUi_hat, 0)
    c[1] += 1j*K[0]*UUi_hat[1]
    c[2] += 1j*K[0]*UUi_hat[2]
    UUi_hat[0] = FFT.fftn(U_dealiased[1]*U_dealiased[1], UUi_hat[0], dealias)
    UUi_hat[1] = FFT.fftn(U_dealiased[1]*U_dealiased[2], UUi_hat[1], dealias)
    UUi_hat[2] = FFT.fftn(U_dealiased[2]*U_dealiased[2], UUi_hat[2], dealias)
    c[1] += (1j*K[1]*UUi_hat[0] + 1j*K[2]*UUi_hat[1])
    c[2] += (1j*K[1]*UUi_hat[1] + 1j*K[2]*UUi_hat[2])
    return c

#@profile
def Cross(a, b, c, dealias=None):
    """c_k = F_k(a x b)"""
    Uc = work[(a, 2)]
    Uc = cross1(Uc, a, b)
    c[0] = FFT.fftn(Uc[0], c[0], dealias)
    c[1] = FFT.fftn(Uc[1], c[1], dealias)
    c[2] = FFT.fftn(Uc[2], c[2], dealias)
    return c

#@profile
def Curl(a, c, dealias=None):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    curl_hat = work[(a, 0)]
    curl_hat = cross2(curl_hat, K, a)
    c[0] = FFT.ifftn(curl_hat[0], c[0], dealias)
    c[1] = FFT.ifftn(curl_hat[1], c[1], dealias)
    c[2] = FFT.ifftn(curl_hat[2], c[2], dealias)    
    return c

def getConvection(convection):
    """Return function used to compute convection"""
    if convection == "Standard":
        
        def Conv(dU):
            U_dealiased = work[((3,)+FFT.work_shape(config.dealias), float, 0)]
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], config.dealias)
            dU = standardConvection(dU, U_dealiased, config.dealias)
            dU[:] *= -1 
            return dU
        
    elif convection == "Divergence":
        
        def Conv(dU):
            U_dealiased = work[((3,)+FFT.work_shape(config.dealias), float, 0)]
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], config.dealias)
            dU = divergenceConvection(dU, U_dealiased, config.dealias, False)
            dU[:] *= -1
            return dU
        
    elif convection == "Skewed":
        
        def Conv(dU):
            U_dealiased = work[((3,)+FFT.work_shape(config.dealias), float, 0)]
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], config.dealias)
            dU = standardConvection(dU, U_dealiased, config.dealias)
            dU = divergenceConvection(dU, U_dealiased, config.dealias, True)
            dU *= -0.5
            return dU
        
    elif convection == "Vortex":
        
        def Conv(dU):
            U_dealiased = work[((3,)+FFT.work_shape(config.dealias), float, 0)]
            curl_dealiased = work[((3,)+FFT.work_shape(config.dealias), float, 1)]
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], config.dealias)
            
            curl_dealiased[:] = Curl(U_hat, curl_dealiased, config.dealias)
            dU = Cross(U_dealiased, curl_dealiased, dU, config.dealias)
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
                            
    dU = conv(dU)
    
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
            U[i] = FFT.ifftn(U_hat[i], U[i])
                 
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
