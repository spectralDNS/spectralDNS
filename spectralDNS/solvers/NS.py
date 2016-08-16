__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
from spectralDNS.mesh.triplyperiodic import setup

vars().update(setup['NS'](**vars()))

def forward_velocity():
    global U_hat, U, FFT
    for i in range(3):
        U_hat[i] = FFT.fftn(U[i], U_hat[i])
    return U_hat
    
def backward_velocity():
    global U, U_hat, FFT
    for i in range(3):
        U[i] = FFT.ifftn(U_hat[i], U[i])
    return U

# Subclass HDF5Writer for appropriate updating of real components
class NSWriter(HDF5Writer):
    
    def update_components(self, U, U_hat, P, P_hat, FFT, params, **kw):
        """Transform to real data when storing the solution"""
        if self.check_if_write(params) or params.tstep % params.checkpoint == 0:
            for i in range(3):
                U[i] = FFT.ifftn(U_hat[i], U[i])
            P = FFT.ifftn(P_hat, P)

hdf5file = NSWriter({'U':U[0], 'V':U[1], 'W':U[2], 'P':P}, 
                     chkpoint={'current':{'U':U, 'P':P}, 'previous':{}},
                     filename=params.solver+'.h5')

def standardConvection(c, U_dealiased, U_hat, dealias=None):
    """c_i = u_j du_i/dx_j"""
    global FFT, work, K
    gradUi = work[(U_dealiased, 2, False)]
    for i in range(3):
        for j in range(3):
            gradUi[j] = FFT.ifftn(1j*K[j]*U_hat[i], gradUi[j], dealias)
        c[i] = FFT.fftn(sum(U_dealiased*gradUi, 0), c[i], dealias)
    return c

def divergenceConvection(c, U_dealiased, dealias=None, add=False):
    """c_i = div(u_i u_j)"""
    global FFT, work, K
    if not add: c.fill(0)
    UUi_hat = work[(c, 0, False)]
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
    global work, cross1, FFT
    Uc = work[(a, 2, False)]
    Uc = cross1(Uc, a, b)
    c[0] = FFT.fftn(Uc[0], c[0], dealias)
    c[1] = FFT.fftn(Uc[1], c[1], dealias)
    c[2] = FFT.fftn(Uc[2], c[2], dealias)
    return c

#@profile
def Curl(a, c, dealias=None):
    """c = curl(a) = F_inv(F(curl(a))) = F_inv(1j*K x a)"""
    global work, cross2, K
    curl_hat = work[(a, 0, False)]
    curl_hat = cross2(curl_hat, K, a)
    c[0] = FFT.ifftn(curl_hat[0], c[0], dealias)
    c[1] = FFT.ifftn(curl_hat[1], c[1], dealias)
    c[2] = FFT.ifftn(curl_hat[2], c[2], dealias)    
    return c

def getConvection(convection):
    """Return function used to compute convection"""
    if convection == "Standard":
        
        def Conv(dU, U_hat):
            global FFT, work, params
            U_dealiased = work[((3,)+FFT.work_shape(params.dealias), float, 0, False)]
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], params.dealias)
            dU = standardConvection(dU, U_dealiased, U_hat, params.dealias)
            dU[:] *= -1 
            return dU
        
    elif convection == "Divergence":
        
        def Conv(dU, U_hat):
            global FFT, work, params
            U_dealiased = work[((3,)+FFT.work_shape(params.dealias), float, 0, False)]
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], params.dealias)
            dU = divergenceConvection(dU, U_dealiased, params.dealias, False)
            dU[:] *= -1
            return dU
        
    elif convection == "Skewed":
        
        def Conv(dU, U_hat):
            global FFT, work, params
            U_dealiased = work[((3,)+FFT.work_shape(params.dealias), float, 0, False)]
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], params.dealias)
            dU = standardConvection(dU, U_dealiased, U_hat, params.dealias)
            dU = divergenceConvection(dU, U_dealiased, params.dealias, True)
            dU *= -0.5
            return dU
        
    elif convection == "Vortex":
        
        #@profile
        def Conv(dU, U_hat):
            global FFT, work, params
            U_dealiased = work[((3,)+FFT.work_shape(params.dealias), float, 0, False)]
            curl_dealiased = work[((3,)+FFT.work_shape(params.dealias), float, 1, False)]
            for i in range(3):
                U_dealiased[i] = FFT.ifftn(U_hat[i], U_dealiased[i], params.dealias)
            
            curl_dealiased[:] = Curl(U_hat, curl_dealiased, params.dealias)
            dU = Cross(U_dealiased, curl_dealiased, dU, params.dealias)
            return dU
        
    return Conv           

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
def ComputeRHS(dU, U_hat):
    """Compute and return entire rhs contribution"""          
    global K2, K, P_hat, K_over_K2, params
    
    dU = conv(dU, U_hat)
    
    dU = add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, params.nu)
        
    return dU

#@profile
def solve():
    global dU, U, U_hat, conv, integrate, profiler, timer
    
    timer = Timer()
    params.t = 0.0
    params.tstep = 0
    integrate = getintegrator(**globals())
    conv = getConvection(params.convection)

    if params.make_profile: profiler = cProfile.Profile()
    
    dt_in = params.dt

    while params.t + params.dt <= params.T+1e-15:
        
        U_hat, params.dt, dt_took = integrate()

        params.t += dt_took
        params.tstep += 1
                 
        update(**globals())

        hdf5file.update(**globals())
        
        timer()
        
        if params.tstep == 1 and params.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()

        #Make sure that the last step hits T exactly.
        if params.t + params.dt >= params.T:
            params.dt = params.T - params.t
            if params.dt <= 1.e-14:
                break

    params.dt = dt_in
    
    dU = ComputeRHS(dU, U_hat)
    
    additional_callback(fU_hat=dU, **globals())

    timer.final(MPI, rank)
    
    if params.make_profile:
        results = create_profile(**globals())
        
    regression_test(**globals())
        
    hdf5file.close()
