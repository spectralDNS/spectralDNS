__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
from ShenDNS import *
from ..shen.Matrices import CDTmat, CTDmat, BDTmat, BTDmat, BTTmat, BTNmat, CNDmat, BNDmat

hdf5file = HDF5Writer(comm, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, config.solver+".h5", 
                      mesh={"x": points, "xp": pointsp, "y": x1, "z": x2})  


# Parameters needed for the pseudo-inverse matrices

# Wavenumbers in y- and z-directions squared 
beta = K[1, 0]**2+K[2, 0]**2

e = zeros(N[0]+3)
d_a = zeros(N[0])
d_b = zeros(N[0])
d2_a = zeros(N[0])
d2_b = zeros(N[0])
d2_c = zeros(N[0])

alpha_k = zeros(N[0])
beta_k  = zeros(N[0])
gamma_k = zeros(N[0])
d_k     = zeros(N[0])

# Wavenumbers in x-direction
kk = arange(N[0]+1).astype(float)
ck = ones(N[0], float)
if SN.quad == "GL": ck[-1] = 2
# Shen coefficients
b_k = -ones(N[0]+1, float)*(kk/(kk+2))**2

for i in xrange(N[0]):
    e[i] = 1
# The non-zero elements of the first pseudo-inverse differentiation matrix    
for k in xrange(1,N[0]):
    d_a[k-1] = -e[k+2]/(2*k)
    d_b[k-1] = ck[k-1]/(2*k)
# The non-zero elements of the second pseudo-inverse differentiation matrix        
for k in xrange(2,N[0]):
    d2_a[k-2] = -e[k+2]/(2*(k**2-1.)) 
    d2_c[k-2] = ck[k-2]/(4*k*(k-1))
    if k<N[0]-1:
        d2_b[k-2] = e[k+4]/(4*k*(k+1))  
# The non-zero elements of the matrix for the pressure correction equation            
for i in xrange(N[0]-3):    
    beta_k[i]  = d2_c[i+1]+ b_k[i+1]*d2_a[i+1]
    if i < (N[0]-5):
        alpha_k[i] = d2_a[i+1] + b_k[i+3]*d2_b[i+1]
        gamma_k[i] = b_k[i+1]*d2_c[i+3]
d_k[:-7] = d2_b[1:-6]

CDT = CDTmat(K[0, :, 0, 0])
CTD = CTDmat(K[0, :, 0, 0])
BDT = BDTmat(K[0, :, 0, 0], ST.quad)
BTD = BTDmat(K[0, :, 0, 0], SN.quad)
BTT = BTTmat(K[0, :, 0, 0], SN.quad)
BTN = BTNmat(K[0, :, 0, 0], SN.quad)
CND = CNDmat(K[0, :, 0, 0])
BND = BNDmat(K[0, :, 0, 0], SN.quad)

dd = BTT.dd.repeat(array(P_hat.shape[1:]).prod()).reshape(P_hat.shape)

#@profile
def pressuregrad(P, P_hat, dU):
    # Pressure gradient x-direction
    dU[0] -= CDT.matvec(P_hat)
    
    # pressure gradient y-direction
    #F_tmp[0] = FST.fss(P, F_tmp[0], ST)
    F_tmp[0] = BDT.matvec(P_hat)
    
    dU[1, :Nu] -= 1j*K[1, :Nu]*F_tmp[0, :Nu]
    
    # pressure gradient z-direction
    dU[2, :Nu] -= 1j*K[2, :Nu]*F_tmp[0, :Nu]    
    
    return dU

def pressuregrad2(Pcorr, dU):
    # Pressure gradient x-direction
    dU[0] -= CDN.matvec(Pcorr)
    
    # pressure gradient y-direction
    F_tmp[0] = BDN.matvec(Pcorr)
    dU[1, :Nu] -= 1j*K[1, :Nu]*F_tmp[0, :Nu]
    
    # pressure gradient z-direction
    dU[2, :Nu] -= 1j*K[2, :Nu]*F_tmp[0, :Nu]    
    
    return dU

def solvePressure(P_hat, Ni):
    """Solve for pressure if Ni is fst of convection"""
    F_tmp[0] = 0
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], F_tmp[0, p_slice])    
    P_hat = HelmholtzSolverP(P_hat, F_tmp[0])
    
    # P in Chebyshev basis for this solver
    P[:] = FST.ifst(P_hat, P, SN)
    P_hat  = FST.fct(P, P_hat, SN)
    P[:] = FST.ifct(P_hat, P, SN)
    P_hat  = FST.fct(P, P_hat, SN)
    return P_hat

def updatepressure(P_hat, Pcorr, U_hat):
    #F_tmp[2] = 0
    #F_tmp[2] = CND.matvec(U_hat[0])
    #F_tmp[2] += 1j*K[1]*BND.matvec(U_hat[1])
    #F_tmp[2] += 1j*K[2]*BND.matvec(U_hat[2])
    #F_tmp[2] = TDMASolverN(F_tmp[2])
    #P_hat += BTN.matvec(Pcorr)/dd
    #P_hat -= nu*BTN.matvec(F_tmp[2])/dd
    
    P_hat += BTN.matvec(Pcorr)/dd
    P_hat -= nu*CTD.matvec(U_hat[0])/dd
    P_hat -= nu*1j*K[1]*BTD.matvec(U_hat[1])/dd
    P_hat -= nu*1j*K[2]*BTD.matvec(U_hat[2])/dd


def PcorrRHS(U_hat, F_tmp):
    """
        RHS of the pressure correction equation
        Pseudo-inverse technic is used
    """
    F_tmp[:] = 0.
    F_tmp2[:] = 0.
    F_tmp2[0] = SFTc.MatVecMult1(d_a, d_b, U_hat[0], F_tmp2[0]) 
    F_tmp2[1] = SFTc.MatVecMult2(d2_a, d2_b,d2_c, U_hat[1], F_tmp2[1]) 
    F_tmp2[2] = SFTc.MatVecMult2(d2_a, d2_b,d2_c, U_hat[2], F_tmp2[2]) 
    
    F_tmp = F_tmp2[0] +1j*K[1]*F_tmp2[1] + 1j*K[2]*F_tmp2[2]
    F_tmp *= 1./dt 
    return F_tmp

def SolvePcorr(F_tmp, Pcorr):
    """
           Solver for pressure correction equation
           Pseudo-inverse technic is used
    """
    Pcorr[:] = 0.0
    Pcorr[1:-2,:,:] = SFTc.PressureSolver(beta,b_k[1:],alpha_k,beta_k,gamma_k,d_k, F_tmp[3:,:,:], Pcorr[1:-2,:,:])
    
    return Pcorr

#@profile

ComputeRHS.func_globals['pressuregrad'] = pressuregrad

pressure_error = zeros(1)
def solve():
    timer = Timer()
    
    while config.t < config.T-1e-8:
        config.t += dt
        config.tstep += 1
        # Tentative momentum solve
        for jj in range(config.velocity_pressure_iters):
            #print "iters: ", jj
            dU[:] = 0
            dU[:] = ComputeRHS(dU, jj)                
            U_hat[0] = HelmholtzSolverU(U_hat[0], dU[0])
            U_hat[1] = HelmholtzSolverU(U_hat[1], dU[1])
            U_hat[2] = HelmholtzSolverU(U_hat[2], dU[2])
        
            # Pressure correction
            F_tmp[0] = PcorrRHS(U_hat, F_tmp[0]) 
            Pcorr[:] = SolvePcorr(F_tmp[0], Pcorr)
            #F_tmp[0] = pressurerhs(U_hat, F_tmp[0]) 
            #Pcorr[:] = HelmholtzSolverP(Pcorr, F_tmp[0])
            
            # Update pressure
            F_tmp[1] = P_hat[:]
            updatepressure(P_hat, Pcorr, U_hat)
            F_tmp[1] -= P_hat

            comm.Allreduce(linalg.norm(Pcorr), pressure_error)
            if jj == 0 and config.print_divergence_progress and rank == 0:
                print "   Divergence error"
            if config.print_divergence_progress:
                if rank == 0:                
                    print "         Pressure correction norm %6d  %2.6e" %(jj, pressure_error[0])
            if pressure_error[0] < config.divergence_tol:
                break
     
        # Update velocity
        dU[:] = 0
        pressuregrad2(Pcorr, dU)        
        dU[0] = TDMASolverD(dU[0])
        dU[1] = TDMASolverD(dU[1])
        dU[2] = TDMASolverD(dU[2])        
        U_hat[:, u_slice] += dt*dU[:, u_slice]  # + since pressuregrad computes negative pressure gradient

        for i in range(3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)
         
        update(**globals())
 
        # Rotate velocities
        U_hat1[:] = U_hat0
        U_hat0[:] = U_hat
        U0[:] = U
        
        P[:] = FST.ifct(P_hat, P, SN)        
        H_hat1[:] = H_hat
        H1[:] = H
                
        timer()
        
        if config.tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
            
    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**globals())
                
    regression_test(**globals())

    hdf5file.close()
