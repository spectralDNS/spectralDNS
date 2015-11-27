__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
from ShenDNS import *
from ..shen.Matrices import CDTmat, CTDmat, BDTmat, BTDmat, BTTmat, BTNmat, CNDmat, BNDmat

hdf5file = HDF5Writer(comm, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, config.solver+".h5", 
                      mesh={"x": points, "xp": pointsp, "y": x1, "z": x2})  

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
def pressuregrad(P_hat, dU):
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

def Divu(U, U_hat, c):
    c[:] = 0
    SFTc.Mult_Div_3D(N[0], K[1, 0], K[2, 0], 
                       U_hat[0, u_slice], U_hat[1, u_slice], U_hat[2, u_slice], c[p_slice])
    c = TDMASolverN(c)
        
    return c

def updatepressure(P_hat, Pcorr, U_hat):
    #F_tmp[2] = 0
    #F_tmp[2] = CND.matvec(U_hat[0])
    #F_tmp[2] += 1j*K[1]*BND.matvec(U_hat[1])
    #F_tmp[2] += 1j*K[2]*BND.matvec(U_hat[2])
    #F_tmp[2] = TDMASolverN(F_tmp[2])
    ##U_tmp[0] = FST.ifst(F_tmp[0], U_tmp[0], SN)
    #P_hat += BTN.matvec(Pcorr)/dd
    #P_hat -= nu*BTN.matvec(F_tmp[2])/dd
    
    P_hat += BTN.matvec(Pcorr)/dd
    P_hat -= nu*CTD.matvec(U_hat[0])/dd
    P_hat -= nu*1j*K[1]*BTD.matvec(U_hat[1])/dd
    P_hat -= nu*1j*K[2]*BTD.matvec(U_hat[2])/dd

#@profile
def solve():
    timer = Timer()
    
    while config.t < config.T-1e-8:
        config.t += dt
        config.tstep += 1
        # Tentative momentum solve
        for jj in range(config.velocity_pressure_iters):
            dU[:] = 0
            dU[:] = ComputeRHS(dU, jj)                
            U_hat[0] = HelmholtzSolverU(U_hat[0], dU[0])
            U_hat[1] = HelmholtzSolverU(U_hat[1], dU[1])
            U_hat[2] = HelmholtzSolverU(U_hat[2], dU[2])
        
            # Pressure correction
            F_tmp[0] = pressurerhs(U_hat, F_tmp[0]) 
            Pcorr[:] = HelmholtzSolverP(Pcorr, F_tmp[0])

            # Update pressure
            updatepressure(P_hat, Pcorr, U_hat)

            if jj == 0 and config.print_divergence_progress:
                print "   Divergence error"
            if config.print_divergence_progress:
                print "         Pressure correction norm %2.6e" %(linalg.norm(Pcorr))
                 
        # Update velocity
        dU[:] = 0
        pressuregrad2(Pcorr, dU)        
        dU[0] = TDMASolverD(dU[0])
        dU[1] = TDMASolverD(dU[1])
        dU[2] = TDMASolverD(dU[2])        
        U_hat[:3, u_slice] += dt*dU[:3, u_slice]  # + since pressuregrad computes negative pressure gradient

        for i in range(3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)
         
        update(**globals())
 
        # Rotate velocities
        U_hat1[:] = U_hat0
        U_hat0[:] = U_hat
        U0[:] = U
        
        P[:] = FST.ifct(P_hat, P, SN)        
        conv1[:] = conv0
                
        timer()
        
        if config.tstep == 1 and config.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
            
    timer.final(MPI, rank)
    
    if config.make_profile:
        results = create_profile(**globals())
                
    regression_test(**globals())

    hdf5file.close()
