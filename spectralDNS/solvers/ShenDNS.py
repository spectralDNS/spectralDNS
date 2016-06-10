__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from spectralinit import *
from spectralDNS.mesh.channel import setup
from ..shen.Matrices import CDNmat, CDDmat, BDNmat, BDDmat, BDTmat, CNDmat, HelmholtzCoeff
from ..shen.la import Helmholtz, TDMA
from ..shen import SFTc

vars().update(setup['IPCS'](**vars()))

assert params.precision == "double"
hdf5file = HDF5Writer(FST, float, {"U":U[0], "V":U[1], "W":U[2], "P":P}, "IPCS.h5", 
                      mesh={"x": x0, "xp": FST.get_mesh_dim(SN, 0), "y": x1, "z": x2})  

HelmholtzSolverU = Helmholtz(params.N[0], sqrt(K[1, 0]**2+K[2, 0]**2+2.0/params.nu/params.dt), ST.quad, False)
HelmholtzSolverP = Helmholtz(params.N[0], sqrt(K[1, 0]**2+K[2, 0]**2), SN.quad, True)
TDMASolverD = TDMA(ST.quad, False)
TDMASolverN = TDMA(SN.quad, True)

alfa = K[1, 0]**2+K[2, 0]**2-2.0/params.nu/params.dt
CDN = CDNmat(K[0, :, 0, 0])
CND = CNDmat(K[0, :, 0, 0])
BDN = BDNmat(K[0, :, 0, 0], ST.quad)
CDD = CDDmat(K[0, :, 0, 0])
BDD = BDDmat(K[0, :, 0, 0], ST.quad)
BDT = BDTmat(K[0, :, 0, 0], SN.quad)
AB = HelmholtzCoeff(K[0, :, 0, 0], -1.0, -alfa, ST.quad)

dpdx = P.copy()
#@profile
def pressuregrad(P, P_hat, dU):
    # Pressure gradient x-direction
    dU[0] -= CDN.matvec(P_hat)
    
    # pressure gradient y-direction
    dP = work[(P_hat, 0)]
    dP = BDN.matvec(P_hat)
    dU[1, :Nu] -= 1j*K[1, :Nu]*dP[:Nu]
    
    # pressure gradient z-direction
    dU[2, :Nu] -= 1j*K[2, :Nu]*dP[:Nu]    
    
    ## Alternatively
    #dpdx[:] = FST.chebDerivative_3D0(P, dpdx, SN)
    #dP = FST.fss(dpdx, dP, ST)
    #dU[0] -= dP
    #dP = FST.fss(P, dP, ST)
    #dU[1, :Nu] -= 1j*K[1, :Nu]*dP[:Nu]
    #dU[2, :Nu] -= 1j*K[2, :Nu]*dP[:Nu]    
    
    return dU

def pressurerhs(U_hat, dU):
    dU[:] = 0.
    SFTc.Mult_Div_3D(params.N[0], K[1, 0], K[2, 0], U_hat[0, u_slice], U_hat[1, u_slice], U_hat[2, u_slice], dU[p_slice])    
    
    #U_tmp2 = work[(U, 0)]
    #F_tmp2 = work[(U_hat, 0)]
    #U_tmp2[0] = Div(U_hat)
    #F_tmp2[0] = FST.fst(U_tmp2[0], F_tmp2[0], ST)
    #U_tmp2[0] = FST.ifst(F_tmp2[0], U_tmp2[0], ST)
    #dU = FST.fss(U_tmp2[0], dU, SN)
    
    dU[p_slice] *= -1./params.dt    
    return dU

def body_force(Sk, dU):
    dU[0, :Nu] -= Sk[0, :Nu]
    dU[1, :Nu] -= Sk[1, :Nu]
    dU[2, :Nu] -= Sk[2, :Nu]
    return dU

def Cross(a, b, c, S):
    Uc = work[(a, 2)]
    Uc[0] = a[1]*b[2]-a[2]*b[1]
    Uc[1] = a[2]*b[0]-a[0]*b[2]
    Uc[2] = a[0]*b[1]-a[1]*b[0]
    c[0] = FST.fss(Uc[0], c[0], S, dealias=params.dealias)
    c[1] = FST.fss(Uc[1], c[1], S, dealias=params.dealias)
    c[2] = FST.fss(Uc[2], c[2], S, dealias=params.dealias)
    return c

def Curl(a, c, S):
    F_tmp = work[(a, 0)]
    Uc = work[(c, 2)]
    SFTc.Mult_CTD_3D_n(params.N[0], a[1], a[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FST.ifct(F_tmp[1], Uc[1], S, dealias=params.dealias)
    dwdx = Uc[2] = FST.ifct(F_tmp[2], Uc[2], S, dealias=params.dealias)
    c[0] = FST.ifst((1j*K[1]*a[2] - 1j*K[2]*a[1]), c[0], S, dealias=params.dealias)
    c[1] = FST.ifst(1j*K[2]*a[0], c[1], S, dealias=params.dealias)
    c[1] -= dwdx
    c[2] = FST.ifst(1j*K[1]*a[0], c[2], S, dealias=params.dealias)
    c[2] *= -1.0
    c[2] += dvdx
    return c

def Div(a_hat):
    Uc_hat = work[(a_hat[0], 0)]
    Uc = work[(U, 2)]
    Uc_hat = CDD.matvec(a_hat[0])
    Uc_hat = TDMASolverD(Uc_hat)    
    dudx = Uc[0] = FST.ifst(Uc_hat, Uc[0], ST) 
    dvdy_h = 1j*K[1]*a_hat[1]
    dvdy = Uc[1] = FST.ifst(dvdy_h, Uc[1], ST)
    dwdz_h = 1j*K[2]*a_hat[2]
    dwdz = Uc[2] = FST.ifst(dwdz_h, Uc[2], ST)
    return dudx+dvdy+dwdz

def Divu(U_hat, c):
    c[:] = 0
    SFTc.Mult_Div_3D(params.N[0], K[1, 0], K[2, 0], U_hat[0, u_slice], 
                     U_hat[1, u_slice], U_hat[2, u_slice], c[p_slice])
    c = TDMASolverN(c)        
    return c

#@profile
def standardConvection(c, U, U_hat):
    c[:] = 0
    Uc = work[(U, 1)]
    Uc2 = work[(U, 2)]
    F_tmp = work[(U_hat, 0)]
    
    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = CDD.matvec(U_hat[0])
    F_tmp[0] = TDMASolverD(F_tmp[0])    
    dudx = Uc[0] = FST.ifst(F_tmp[0], Uc[0], ST, dealias=params.dealias)   
    
    #F_tmp[0] = CND.matvec(U_hat[0])
    #F_tmp[0] = TDMASolverN(F_tmp[0])    
    #quad = SN.quad
    #SN.quad = ST.quad
    #dudx = Uc[0] = FST.ifst(F_tmp[0], Uc[0], SN, dealias=params.dealias)       
    
    SFTc.Mult_CTD_3D_n(params.N[0], U_hat[1], U_hat[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FST.ifct(F_tmp[1], Uc[1], ST, dealias=params.dealias)
    dwdx = Uc[2] = FST.ifct(F_tmp[2], Uc[2], ST, dealias=params.dealias)
    
    #dudx = U_tmp[0] = chebDerivative_3D0(U[0], U_tmp[0])
    #dvdx = U_tmp[1] = chebDerivative_3D0(U[1], U_tmp[1])
    #dwdx = U_tmp[2] = chebDerivative_3D0(U[2], U_tmp[2])    
    
    dudy = Uc2[0] = FST.ifst(1j*K[1]*U_hat[0], Uc2[0], ST, dealias=params.dealias)    
    dudz = Uc2[1] = FST.ifst(1j*K[2]*U_hat[0], Uc2[1], ST, dealias=params.dealias)
    c[0] = FST.fss(U[0]*dudx + U[1]*dudy + U[2]*dudz, c[0], ST, dealias=params.dealias)
    
    Uc2[:] = 0    
    dvdy = Uc2[0] = FST.ifst(1j*K[1]*U_hat[1], Uc2[0], ST, dealias=params.dealias)
    #F_tmp[0] = FST.fst(U[1], F_tmp[0], SN)
    #dvdy_h = 1j*K[1]*F_tmp[0]    
    #dvdy = U_tmp2[0] = FST.ifst(dvdy_h, U_tmp2[0], SN)
    ##########
    
    dvdz = Uc2[1] = FST.ifst(1j*K[2]*U_hat[1], Uc2[1], ST, dealias=params.dealias)
    c[1] = FST.fss(U[0]*dvdx + U[1]*dvdy + U[2]*dvdz, c[1], ST, dealias=params.dealias)
    
    Uc2[:] = 0
    dwdy = Uc2[0] = FST.ifst(1j*K[1]*U_hat[2], Uc2[0], ST, dealias=params.dealias)
    
    dwdz = Uc2[1] = FST.ifst(1j*K[2]*U_hat[2], Uc2[1], ST, dealias=params.dealias)
    
    #F_tmp[0] = FST.fst(U[2], F_tmp[0], SN)
    #dwdz_h = 1j*K[2]*F_tmp[0]    
    #dwdz = U_tmp2[1] = FST.ifst(dwdz_h, U_tmp2[1], SN)    
    #########
    c[2] = FST.fss(U[0]*dwdx + U[1]*dwdy + U[2]*dwdz, c[2], ST, dealias=params.dealias)
    
    # Reset
    #SN.quad = quad

    return c

def divergenceConvection(c, U, U_hat, add=False):
    """c_i = div(u_i u_j)"""
    if not add: c.fill(0)
    F_tmp = work[(U_hat, 0)]
    F_tmp2 = work[(U_hat, 1)]

    #U_tmp[0] = chebDerivative_3D0(U[0]*U[0], U_tmp[0])
    #U_tmp[1] = chebDerivative_3D0(U[0]*U[1], U_tmp[1])
    #U_tmp[2] = chebDerivative_3D0(U[0]*U[2], U_tmp[2])
    #c[0] = fss(U_tmp[0], c[0], ST)
    #c[1] = fss(U_tmp[1], c[1], ST)
    #c[2] = fss(U_tmp[2], c[2], ST)
    
    F_tmp[0] = FST.fst(U[0]*U[0], F_tmp[0], ST, dealias=params.dealias)
    F_tmp[1] = FST.fst(U[0]*U[1], F_tmp[1], ST, dealias=params.dealias)
    F_tmp[2] = FST.fst(U[0]*U[2], F_tmp[2], ST, dealias=params.dealias)
    
    c[0] += CDD.matvec(F_tmp[0])
    c[1] += CDD.matvec(F_tmp[1])
    c[2] += CDD.matvec(F_tmp[2])
    
    F_tmp2[0] = FST.fss(U[0]*U[1], F_tmp2[0], ST, dealias=params.dealias)
    F_tmp2[1] = FST.fss(U[0]*U[2], F_tmp2[1], ST, dealias=params.dealias)    
    c[0] += 1j*K[1]*F_tmp2[0] # duvdy
    c[0] += 1j*K[2]*F_tmp2[1] # duwdz
    
    F_tmp[0] = FST.fss(U[1]*U[1], F_tmp[0], ST, dealias=params.dealias)
    F_tmp[1] = FST.fss(U[1]*U[2], F_tmp[1], ST, dealias=params.dealias)
    F_tmp[2] = FST.fss(U[2]*U[2], F_tmp[2], ST, dealias=params.dealias)
    c[1] += 1j*K[1]*F_tmp[0]  # dvvdy
    c[1] += 1j*K[2]*F_tmp[1]  # dvwdz  
    c[2] += 1j*K[1]*F_tmp[1]  # dvwdy
    c[2] += 1j*K[2]*F_tmp[2]  # dwwdz
    return c    

def getConvection(convection):
    if convection == "Standard":
        
        def Conv(H_hat, U, U_hat):
            
            U_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            for i in range(3):
                U_dealiased[i] = FST.ifst(U_hat[i], U_dealiased[i], ST, params.dealias)

            H_hat = standardConvection(H_hat, U_dealiased, U_hat)
            H_hat[:] *= -1
            return H_hat
        
    elif convection == "Divergence":
        
        def Conv(H_hat, U, U_hat):
            
            U_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            for i in range(3):
                U_dealiased[i] = FST.ifst(U_hat[i], U_dealiased[i], ST, params.dealias)

            H_hat = divergenceConvection(H_hat, U_dealiased, U_hat, False)
            H_hat[:] *= -1
            return H_hat
        
    elif convection == "Skew":
        
        def Conv(H_hat, U, U_hat):
            
            U_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            for i in range(3):
                U_dealiased[i] = FST.ifst(U_hat[i], U_dealiased[i], ST, params.dealias)

            H_hat = standardConvection(H_hat, U_dealiased, U_hat)
            H_hat = divergenceConvection(H_hat, U_dealiased, U_hat, True)            
            H_hat *= -0.5
            return H_hat

    elif convection == "Vortex":
        
        def Conv(H_hat, U, U_hat):
            
            U_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            curl_dealiased = work[((3,)+FST.work_shape(params.dealias), float, 1)]
            for i in range(3):
                U_dealiased[i] = FST.ifst(U_hat[i], U_dealiased[i], ST, params.dealias)
            
            curl_dealiased[:] = Curl(U_hat, curl_dealiased, ST)
            H_hat[:] = Cross(U_dealiased, curl_dealiased, H_hat, ST)            
            return H_hat
        
    return Conv           

conv = getConvection(params.convection)
    
#@profile
def ComputeRHS(dU, jj):
    global H_hat, conv
    
    conv = getConvection(params.convection)

    # Add convection to rhs
    if jj == 0:
        H_hat = conv(H_hat, U0, U_hat0)
        
        # Compute diffusion
        diff0[:] = 0
        diff0[0] = AB.matvec(U_hat0[0], diff0[0])
        diff0[1] = AB.matvec(U_hat0[1], diff0[1])
        diff0[2] = AB.matvec(U_hat0[2], diff0[2])
    
    H_hat0[:] = 1.5*H_hat - 0.5*H_hat1

    dU[:] = H_hat0
    
    # Add pressure gradient and body force
    dU = pressuregrad(P, P_hat, dU)
    dU = body_force(Sk, dU)
    
    # Scale by 2/nu factor
    dU[:] *= 2./params.nu
    
    # Add diffusion
    dU[:] += diff0
        
    return dU

def solvePressure(P_hat, Ni):
    """Solve for pressure if Ni is fst of convection"""
    dP = work[(P_hat, 0)]
    SFTc.Mult_Div_3D(params.N[0], K[1, 0], K[2, 0], Ni[0, u_slice], Ni[1, u_slice], Ni[2, u_slice], dP[p_slice])    
    P_hat = HelmholtzSolverP(P_hat, dP)
    return P_hat

#@profile
pressure_error = zeros(1)
def solve():
    global dU, U_hat
    
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
            P_hat[p_slice] += Pcorr[p_slice]
            
            comm.Allreduce(linalg.norm(Pcorr), pressure_error)
            if jj == 0 and params.print_divergence_progress and rank == 0:
                print "   Divergence error"
            if params.print_divergence_progress:
                if rank == 0:                
                    print "         Pressure correction norm %6d  %2.6e" %(jj, pressure_error[0])
            if pressure_error[0] < params.divergence_tol:
                break
            
        for i in range(3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)
                 
        # Update velocity
        dU[:] = 0
        Uc = work[(U[0], 0)]
        Uc = FST.ifst(Pcorr, Uc, SN)
        pressuregrad(Uc, Pcorr, dU)
        dU[0] = TDMASolverD(dU[0])
        dU[1] = TDMASolverD(dU[1])
        dU[2] = TDMASolverD(dU[2])
        U_hat[:, u_slice] += params.dt*dU[:, u_slice]  # + since pressuregrad computes negative pressure gradient

        for i in range(3):
            U[i] = FST.ifst(U_hat[i], U[i], ST)

        update(**globals())
 
        # Rotate velocities
        U_hat1[:] = U_hat0
        U_hat0[:] = U_hat
        U0[:] = U
        
        P[:] = FST.ifst(P_hat, P, SN)        
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
