__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .IPCS import *
from ..shen.Matrices import CDTmat, CTDmat, BDTmat, BTDmat, BTTmat, BTNmat, CNDmat, BNDmat

def get_context():
    """Set up context for solver"""

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad=params.Dquad, threads=params.threads,
                            planner_effort=params.planner_effort["dct"])
    SN = ShenNeumannBasis(quad=params.Nquad, threads=params.threads,
                          planner_effort=params.planner_effort["dct"])

    Nf = params.N[2]/2+1 # Number of independent complex wavenumbers in z-direction
    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nq = params.N[0]-3   # Number of pressure modes in Shen basis
    u_slice = slice(0, Nu)
    p_slice = slice(1, Nu)

    FST = SlabShen_R2C(params.N, params.L, comm, threads=params.threads,
                       communication=params.communication,
                       planner_effort=params.planner_effort,
                       dealias_cheb=params.dealias_cheb)

    float, complex, mpitype = datatypes("double")

    # Get grid for velocity points
    X = FST.get_local_mesh(ST)
    x0, x1, x2 = FST.get_mesh_dims(ST)

    U     = zeros((3,)+FST.real_shape(), dtype=float)
    U_hat = zeros((3,)+FST.complex_shape(), dtype=complex)
    P     = zeros(FST.real_shape(), dtype=float)
    P_hat = zeros(FST.complex_shape(), dtype=complex)
    Pcorr = zeros(FST.complex_shape(), dtype=complex)
    U0      = zeros((3,)+FST.real_shape(), dtype=float)
    U_hat0  = zeros((3,)+FST.complex_shape(), dtype=complex)
    U_hat1  = zeros((3,)+FST.complex_shape(), dtype=complex)
    dU      = zeros((3,)+FST.complex_shape(), dtype=complex)
    H_hat    = zeros((3,)+FST.complex_shape(), dtype=complex)
    H_hat0   = zeros((3,)+FST.complex_shape(), dtype=complex)
    H_hat1   = zeros((3,)+FST.complex_shape(), dtype=complex)

    diff0   = zeros((3,)+FST.complex_shape(), dtype=complex)
    Source  = zeros((3,)+FST.real_shape(), dtype=float)
    Sk      = zeros((3,)+FST.complex_shape(), dtype=complex)

    K = FST.get_scaled_local_wavenumbermesh()
    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)
    work = work_arrays()

    # Primary variable
    u = (U_hat, P_hat)

    nu, dt, N = params.nu, params.dt, params.N

    # Collect all linear algebra solvers
    la = config.AttributeDict(dict(
        HelmholtzSolverU = Helmholtz(N[0], np.sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/dt), ST),
        HelmholtzSolverP = Helmholtz(N[0], np.sqrt(K[1, 0]**2+K[2, 0]**2), SN),
        TDMASolverD = TDMA(ST),
        TDMASolverN = TDMA(SN)
        )
    )

    alfa = K[1, 0]**2+K[2, 0]**2-2.0/nu/dt

    # Collect all matrices
    kx = K[0, :, 0, 0]
    mat = config.AttributeDict(dict(
        CDN = CDNmat(kx),
        CND = CNDmat(kx),
        BDN = BDNmat(kx, ST.quad),
        CDD = CDDmat(kx),
        BDD = BDDmat(kx, ST.quad),
        AB = HelmholtzCoeff(kx, -1.0, -alfa, ST.quad),
        CDT = CDTmat(kx),
        CTD = CTDmat(kx),
        BDT = BDTmat(kx, ST.quad),
        BTD = BTDmat(kx, SN.quad),
        BTT = BTTmat(kx, SN.quad),
        BTN = BTNmat(kx, SN.quad),
        BND = BNDmat(kx, SN.quad)
        )
    )

    dd = mat.BTT[0].repeat(np.array(P_hat.shape[1:]).prod()).reshape(P_hat.shape)


    hdf5file = IPCSRWriter({"U":U[0], "V":U[1], "W":U[2], "P":P},
                           chkpoint={'current':{'U':U, 'P':P}, 'previous':{'U':U0}},
                           filename=params.solver+".h5",
                           mesh={"x": x0, "xp": FST.get_mesh_dim(SN, 0), "y": x1, "z": x2})


    return config.AttributeDict(locals())


class IPCSRWriter(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        U = get_velocity(**context)
        P = get_pressure(**context)
        if params.tstep % params.checkpoint == 0:
            c = config.AttributeDict(context)
            U0 = get_velocity(c.U0, c.U_hat0, c.FST, c.ST)


def get_pressure(P, P_hat, FST, SN, **context):
    """Compute pressure from context"""
    P = FST.ifct(P_hat, P, SN.CT)
    return P

def set_pressure(P_hat, P, FST, SN, **context):
    """Compute pressure from context"""
    P_hat = FST.fct(P, P_hat, SN.CT)
    return P_hat

def pressuregrad(rhs, p_hat, mat, work, K, Nu):
    """Compute contribution to rhs from pressure gradient

    Overload because pressure has different space in IPCSR

    """
    w0 = work[(p_hat, 0, False)]
    # Pressure gradient x-direction
    rhs[0] -= mat.CDT.matvec(p_hat, w0)

    # pressure gradient y-direction
    dP = work[(p_hat, 1, False)]
    #dP = FST.fss(P, dP, ST)
    dP = mat.BDT.matvec(p_hat, dP)

    rhs[1, :Nu] -= 1j*K[1, :Nu]*dP[:Nu]

    # pressure gradient z-direction
    rhs[2, :Nu] -= 1j*K[2, :Nu]*dP[:Nu]

    return rhs

def pressuregrad2(rhs, p_corr, K, mat, work, Nu):
    dP = work[(p_corr, 0, False)]

    # Pressure gradient x-direction
    rhs[0] -= mat.CDN.matvec(p_corr, dP)

    # pressure gradient y-direction

    dP = mat.BDN.matvec(p_corr, dP)
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
    LUsolve.Mult_Div_3D(params.N[0], K[1, 0], K[2, 0], H_hat[0, u_slice],
                     H_hat[1, u_slice], H_hat[2, u_slice], F_tmp[p_slice])
    P_hat = la.HelmholtzSolverP(P_hat, F_tmp)

    # P in Chebyshev basis for this solver
    P[:] = FST.ifst(P_hat, P, SN)
    P_hat  = FST.fct(P, P_hat, SN.CT)
    P[:] = FST.ifct(P_hat, P, SN.CT)
    P_hat  = FST.fct(P, P_hat, SN.CT)
    return P_hat

def updatepressure(p_hat, p_corr, u_hat, K, mat, dd, work):
    #F_tmp = work[(P_hat, 0)]
    #F_tmp[:] = CND.matvec(U_hat[0])
    #F_tmp += 1j*K[1]*BND.matvec(U_hat[1])
    #F_tmp += 1j*K[2]*BND.matvec(U_hat[2])
    #F_tmp = TDMASolverN(F_tmp)
    #P_hat += BTN.matvec(Pcorr)/dd
    #P_hat -= nu*BTN.matvec(F_tmp)/dd

    w0 = work[(p_corr, 0, False)]
    p_hat += mat.BTN.matvec(p_corr, w0)/dd
    p_hat -= params.nu*mat.CTD.matvec(u_hat[0], w0)/dd
    p_hat -= params.nu*1j*K[1]*mat.BTD.matvec(u_hat[1], w0)/dd
    p_hat -= params.nu*1j*K[2]*mat.BTD.matvec(u_hat[2], w0)/dd
    return p_hat

def solve_pressure_correction(p_hat, u_hat, solver,
                              Pcorr, K, mat, dd, la, work, u_slice, p_slice, **context):
    dP = work[(p_hat, 0)]
    dP = solver.pressurerhs(dP, u_hat, K, u_slice, p_slice)
    Pcorr[:] = la.HelmholtzSolverP(Pcorr, dP)
    # Update pressure
    p_hat = updatepressure(p_hat, Pcorr, u_hat, K, mat, dd, work)
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
