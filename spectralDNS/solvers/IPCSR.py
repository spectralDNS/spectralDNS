__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from IPCS import *
from ..shen.Matrices import CDTmat, CTDmat, BDTmat, BTDmat, BTTmat, BTNmat, CNDmat, BNDmat

get_context_IPCS = get_context
def get_context():
    d = get_context_IPCS()
    
    k = d.K[0, :, 0, 0] 
    d.mat.update(dict(
        CDT = CDTmat(k),
        CTD = CTDmat(k),
        BDT = BDTmat(k, d.ST.quad),
        BTD = BTDmat(k, d.SN.quad),
        BTT = BTTmat(k, d.SN.quad),
        BTN = BTNmat(k, d.SN.quad),
        CND = CNDmat(k),
        BND = BNDmat(k, d.SN.quad)
        )
    )

    d.dd = d.mat.BTT.dd.repeat(np.array(d.P_hat.shape[1:]).prod()).reshape(d.P_hat.shape)

    return d

def get_pressure(P, P_hat, FST, SN, **context):
    """Compute pressure from context"""
    P = FST.ifct(P_hat, P, SN)
    return P

def set_pressure(P_hat, P, FST, SN, **context):
    """Compute pressure from context"""
    P_hat = FST.fct(P, P_hat, SN)
    return P_hat

def pressuregrad(rhs, p_hat, mat, work, K, Nu):
    """Compute contribution to rhs from pressure gradient
    
    Overload because pressure has different space in IPCSR
    
    """
    
    # Pressure gradient x-direction
    rhs[0] -= mat.CDT.matvec(p_hat)
    
    # pressure gradient y-direction
    dP = work[(p_hat, 0)]
    #dP = FST.fss(P, dP, ST)
    dP[:] = mat.BDT.matvec(p_hat)
    
    rhs[1, :Nu] -= 1j*K[1, :Nu]*dP[:Nu]
    
    # pressure gradient z-direction
    rhs[2, :Nu] -= 1j*K[2, :Nu]*dP[:Nu]   
    
    return rhs

def pressuregrad2(rhs, p_corr, K, mat, work, Nu):
    # Pressure gradient x-direction
    rhs[0] -= mat.CDN.matvec(p_corr)
    
    # pressure gradient y-direction
    dP = work[(p_corr, 0)]
    dP[:] = mat.BDN.matvec(p_corr)
    rhs[1, :Nu] -= 1j*K[1, :Nu]*dP[:Nu]
    
    # pressure gradient z-direction
    rhs[2, :Nu] -= 1j*K[2, :Nu]*dP[:Nu]    
    
    return rhs

def compute_pressure(P_hat, H_hat, U_hat, U_hat0, K, FST, ST, work, mat, la,
                     u_slice, p_slice, P, SN, **context):
    """Solve for pressure if Ni is fst of convection"""
    conv = getConvection(params.convection)
    H_hat = conv(H_hat, 0.5*(U_hat+U_hat0), K, FST, ST, work, mat, la)
    for i in range(3):
        H_hat[i] = la.TDMASolverD(H_hat[i])
    H_hat *= -1
    
    F_tmp = work[(P_hat, 0)] 
    SFTc.Mult_Div_3D(params.N[0], K[1, 0], K[2, 0], H_hat[0, u_slice],
                     H_hat[1, u_slice], H_hat[2, u_slice], F_tmp[p_slice])
    P_hat = la.HelmholtzSolverP(P_hat, F_tmp)
    
    # P in Chebyshev basis for this solver
    P[:] = FST.ifst(P_hat, P, SN)
    P_hat  = FST.fct(P, P_hat, SN)
    P[:] = FST.ifct(P_hat, P, SN)
    P_hat  = FST.fct(P, P_hat, SN)
    return P_hat

def updatepressure(p_hat, p_corr, u_hat, K, mat, dd):
    #F_tmp = work[(P_hat, 0)]
    #F_tmp[:] = CND.matvec(U_hat[0])
    #F_tmp += 1j*K[1]*BND.matvec(U_hat[1])
    #F_tmp += 1j*K[2]*BND.matvec(U_hat[2])
    #F_tmp = TDMASolverN(F_tmp)
    #P_hat += BTN.matvec(Pcorr)/dd
    #P_hat -= nu*BTN.matvec(F_tmp)/dd
    
    p_hat += mat.BTN.matvec(p_corr)/dd
    p_hat -= params.nu*mat.CTD.matvec(u_hat[0])/dd
    p_hat -= params.nu*1j*K[1]*mat.BTD.matvec(u_hat[1])/dd
    p_hat -= params.nu*1j*K[2]*mat.BTD.matvec(u_hat[2])/dd
    return p_hat

def solve_pressure_correction(p_hat, u_hat, solver,
                              Pcorr, K, mat, dd, la, work, u_slice, p_slice, **context):
    dP = work[(p_hat, 0)]
    dP = solver.pressurerhs(dP, u_hat, K, u_slice, p_slice)
    Pcorr[:] = la.HelmholtzSolverP(Pcorr, dP)
    # Update pressure    
    p_hat = updatepressure(p_hat, Pcorr, u_hat, K, mat, dd)    
    return p_hat, Pcorr

def update_velocity(u_hat, p_corr, rhs, solver,
                    K, mat, work, la, Nu, u_slice, p_slice, **context):
    rhs[:] = 0
    rhs = solver.pressuregrad2(rhs, p_corr, K, mat, work, Nu)
    rhs[0] = la.TDMASolverD(rhs[0])
    rhs[1] = la.TDMASolverD(rhs[1])
    rhs[2] = la.TDMASolverD(rhs[2])        
    u_hat[:, u_slice] += params.dt*rhs[:, u_slice]  # + since pressuregrad computes negative pressure gradient
    return u_hat
