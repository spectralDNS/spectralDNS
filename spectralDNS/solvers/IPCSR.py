__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from IPCS import *
from ..shen.Matrices import CDTmat, CTDmat, BDTmat, BTDmat, BTTmat, BTNmat, CNDmat, BNDmat

# Get and update the global namespace of the ShenDNS solver (to avoid having two namespaces filled with arrays)
# Overload just a few routines
context = solve.func_globals
context.update(setup['IPCSR'](**vars()))
vars().update(context)

hdf5file = HDF5Writer({"U":U[0], "V":U[1], "W":U[2], "P":P}, 
                      chkpoint={'current':{'U':U, 'P':P}, 'previous':{'U':U0}},
                      filename=params.solver+".h5", 
                      mesh={"x": x0, "xp": FST.get_mesh_dim(SN, 0), "y": x1, "z": x2})

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
    dP = work[(P_hat, 0)]
    #dP = FST.fss(P, dP, ST)
    dP[:] = BDT.matvec(P_hat)
    
    dU[1, :Nu] -= 1j*K[1, :Nu]*dP[:Nu]
    
    # pressure gradient z-direction
    dU[2, :Nu] -= 1j*K[2, :Nu]*dP[:Nu]    
    
    return dU

def pressuregrad2(Pcorr, dU):
    # Pressure gradient x-direction
    dU[0] -= CDN.matvec(Pcorr)
    
    # pressure gradient y-direction
    dP = work[(P_hat, 0)]
    dP[:] = BDN.matvec(Pcorr)
    dU[1, :Nu] -= 1j*K[1, :Nu]*dP[:Nu]
    
    # pressure gradient z-direction
    dU[2, :Nu] -= 1j*K[2, :Nu]*dP[:Nu]    
    
    return dU

def solvePressure(P_hat, Ni):
    """Solve for pressure if Ni is fst of convection"""
    F_tmp = work[(P_hat, 0)] 
    SFTc.Mult_Div_3D(params.N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], F_tmp[p_slice])    
    P_hat = HelmholtzSolverP(P_hat, F_tmp)
    
    # P in Chebyshev basis for this solver
    P[:] = FST.ifst(P_hat, P, SN)
    P_hat  = FST.fct(P, P_hat, SN)
    P[:] = FST.ifct(P_hat, P, SN)
    P_hat  = FST.fct(P, P_hat, SN)
    return P_hat

def updatepressure(P_hat, Pcorr, U_hat):
    #F_tmp = work[(P_hat, 0)]
    #F_tmp[:] = CND.matvec(U_hat[0])
    #F_tmp += 1j*K[1]*BND.matvec(U_hat[1])
    #F_tmp += 1j*K[2]*BND.matvec(U_hat[2])
    #F_tmp = TDMASolverN(F_tmp)
    #P_hat += BTN.matvec(Pcorr)/dd
    #P_hat -= nu*BTN.matvec(F_tmp)/dd
    
    P_hat += BTN.matvec(Pcorr)/dd
    P_hat -= params.nu*CTD.matvec(U_hat[0])/dd
    P_hat -= params.nu*1j*K[1]*BTD.matvec(U_hat[1])/dd
    P_hat -= params.nu*1j*K[2]*BTD.matvec(U_hat[2])/dd

# Update ComputeRHS to use current pressuregrad
ComputeRHS.func_globals['pressuregrad'] = pressuregrad

#@profile
pressure_error = zeros(1)
def solve():
    timer = Timer()
    
    while params.t < params.T-1e-8:
        params.t += params.dt
        params.tstep += 1
        # Tentative momentum solve
        for jj in range(params.velocity_pressure_iters):
            dU[:] = 0
            dU[:] = ComputeRHS(dU, jj)                
            U_hat[0] = HelmholtzSolverU(U_hat[0], dU[0])
            U_hat[1] = HelmholtzSolverU(U_hat[1], dU[1])
            U_hat[2] = HelmholtzSolverU(U_hat[2], dU[2])
        
            # Pressure correction
            dP = work[(P_hat, 0)]
            dP = pressurerhs(U_hat, dP) 
            Pcorr[:] = HelmholtzSolverP(Pcorr, dP)

            # Update pressure            
            dP[:] = P_hat
            updatepressure(P_hat, Pcorr, U_hat)
            dP -= P_hat

            comm.Allreduce(linalg.norm(Pcorr), pressure_error)
            if jj == 0 and params.print_divergence_progress and rank == 0:
                print "   Divergence error"
            if params.print_divergence_progress:
                if rank == 0:                
                    print "         Pressure correction norm %6d  %2.6e" %(jj, pressure_error[0])
            if pressure_error[0] < params.divergence_tol:
                break
     
        # Update velocity
        dU[:] = 0
        pressuregrad2(Pcorr, dU)        
        dU[0] = TDMASolverD(dU[0])
        dU[1] = TDMASolverD(dU[1])
        dU[2] = TDMASolverD(dU[2])        
        U_hat[:, u_slice] += params.dt*dU[:, u_slice]  # + since pressuregrad computes negative pressure gradient

        update(**globals())

        hdf5file.update(**globals())
          
        # Rotate velocities
        U_hat1[:] = U_hat0
        U_hat0[:] = U_hat
        
        #P[:] = FST.ifct(P_hat, P, SN)        
        H_hat1[:] = H_hat
                
        timer()
        
        if params.tstep == 1 and params.make_profile:
            #Enable profiling after first step is finished
            profiler.enable()
            
    timer.final(MPI, rank)
    
    if params.make_profile:
        results = create_profile(**globals())
                
    regression_test(**globals())

    hdf5file.close()
