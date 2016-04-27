__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ShenKMM import *

a = (8./15., 5./12., 3./4.)
b = (0.0, -17./60., -5./12.)

HelmholtzSolverG = [Helmholtz(N[0], sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/(a[rk]+b[rk])/dt), 
                              ST.quad, False) for rk in range(3)]

BiharmonicSolverU = [Biharmonic(N[0], -nu*(a[rk]+b[rk])*dt/2., 1.+nu*(a[rk]+b[rk])*dt*K2[0], 
                                -(K2[0] + nu*(a[rk]+b[rk])*dt/2.*K4[0]), SB.quad) for rk in range(3)]

HelmholtzSolverU0 = [Helmholtz(N[0], sqrt(2./nu/(a[rk]+b[rk])/dt), ST.quad, False) for rk in range(3)]

AC = [BiharmonicCoeff(K[0, :, 0, 0], nu*(a[rk]+b[rk])*dt/2., (1. - nu*(a[rk]+b[rk])*dt*K2[0]),
                      -(K2[0] - nu*(a[rk]+b[rk])*dt/2.*K4[0]), SB.quad) for rk in range(3)]

AB = [HelmholtzCoeff(K[0, :, 0, 0], -1.0, -(K2[0] - 2.0/nu/dt/(a[rk]+b[rk])), ST.quad) for rk in range(3)]

U_hat1 = U_hat0.copy()
U_hat2 = U_hat0.copy()
hg0 = hg.copy()
hv0 = hv.copy()
u0_hat = zeros((3, N[0]), dtype=complex)
h0_hat = zeros((3, N[0]), dtype=complex)
h0 = zeros((2, N[0]), dtype=complex)
h1 = zeros((2, N[0]), dtype=complex)

@optimizer
def add_diffusion_u(u, d, AC, SBB, ABB, BBB, nu, dt, K2, K4, a, b):
    d[:] = nu*(a+b)*dt/2.*SBB.matvec(u)
    d += (1. - nu*(a+b)*dt*K2)*ABB.matvec(u)
    d -= (K2 - nu*(a+b)*dt/2.*K2**2)*BBB.matvec(u)
    return d

def RKstep(U_hat, g, dU, rk):
    global conv1, hv, hg, hv0, hg0, a, b, h0, h1
    
    # Compute convection
    H_hat[:] = conv(H_hat, U, U_hat)    
    
    # Compute diffusion for g and u-equation
    diff0[1] = AB[rk].matvec(g, diff0[1])
    
    # Compute diffusion++ for u-equation
    diff0[0] = add_diffusion_u(U_hat[0], diff0[0], AC[rk], SBB, ABB, BBB, nu, dt, K2, K4, a[rk], b[rk])
    
    #hv[:] = -K2*BBD.matvec(H_hat[0])
    hv[:] = FST.fss(H[0], hv, SB)
    hv *= -K2
    hv -= 1j*K[1]*CBD.matvec(H_hat[1])
    hv -= 1j*K[2]*CBD.matvec(H_hat[2])    
    hg[:] = 1j*K[1]*BDD.matvec(H_hat[2]) - 1j*K[2]*BDD.matvec(H_hat[1])
    
    dU[0] = (hv*a[rk] + hv0*b[rk])*dt + diff0[0]
    dU[1] = (hg*a[rk] + hg0*b[rk])*2./nu/(a[rk]+b[rk]) + diff0[1]
    
    U_hat[0] = BiharmonicSolverU[rk](U_hat[0], dU[0])
    g[:] = HelmholtzSolverG[rk](g, dU[1])

    f_hat = work[(U_hat[0], 0)]
    f_hat = - CDB.matvec(U_hat[0])
    f_hat = TDMASolverD(f_hat)

    u0_hat[1, :] = U_hat[1, :, 0, 0]
    u0_hat[2, :] = U_hat[2, :, 0, 0]
    
    U_hat[1] = -1j*(K_over_K2[1]*f_hat - K_over_K2[2]*g)
    U_hat[2] = -1j*(K_over_K2[2]*f_hat + K_over_K2[1]*g) 

    # Remains to fix wavenumber 0    
    if rank == 0:
        h0_hat[1, :] = H_hat[1, :, 0, 0]
        h0_hat[2, :] = H_hat[2, :, 0, 0]
        
        h1[0] = BDD.matvec(h0_hat[1])
        h1[1] = BDD.matvec(h0_hat[2])
        h1[0] -= Sk[1, :, 0, 0]  # Subtract constant pressure gradient
        
        beta = 2./nu/(a[rk]+b[rk])/dt
        w = beta*(a[rk]*h1[0] + b[rk]*h0[0])*dt
        w -= ADD.matvec(u0_hat[1])
        w += beta*BDD.matvec(u0_hat[1])    
        u0_hat[1] = HelmholtzSolverU0[rk](u0_hat[1], w)
        
        w = beta*(a[rk]*h1[1] + b[rk]*h0[1])*dt
        w -= ADD.matvec(u0_hat[2])
        w += beta*BDD.matvec(u0_hat[2])    
        u0_hat[2] = HelmholtzSolverU0[rk](u0_hat[2], w)
            
        U_hat[1, :, 0, 0] = u0_hat[1]
        U_hat[2, :, 0, 0] = u0_hat[2]
    
    return U_hat

def regression_test(**kw):
    pass

#@profile
def solve():
    timer = Timer()
    
    while config.t < config.T-1e-10:
        config.t += dt
        config.tstep += 1

        dU[:] = 0
        hv0[:] = 0
        hg0[:] = 0
        h0[:] = 0
        for rk in range(3):            
            U_hat[:] = RKstep(U_hat, g, dU, rk)
            hv0[:] = hv
            hg0[:] = hg
            h0[:]  = h1
        
        update(**globals())
                 
        timer()
        
        if config.tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
            
    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**globals())
                
    regression_test(**globals())

    hdf5file.close()
