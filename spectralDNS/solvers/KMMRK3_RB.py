__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2018-08-29"
__copyright__ = "Copyright (C) 2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=function-redefined,unbalanced-tuple-unpacking,unused-variable,unused-argument

from .KMMRK3 import *
from .spectralinit import end_of_tstep

KMMRK3_context = get_context
KMMRK3_ComputeRHS = ComputeRHS
KMMRK3_solve_linear = solve_linear

def get_context():
    c = KMMRK3_context()

    c.RB = RB = Basis(config.params.N[0], 'C', bc=(0, 1))
    c.FRB = FRB = TensorProductSpace(comm, (RB, c.K0, c.K1), **c.kw0)
    c.FRBp = FRBp = TensorProductSpace(comm, (RB, c.K0p, c.K1p), **c.kw0)

    c.dU = Function(c.VFS)  # rhs vector for integrator. Now three components, not two
    c.phi = Array(FRB)
    c.phi0 = Array(FRBp)
    c.phi_hat = Function(FRB)
    c.Ua = Array(c.VFSp)
    c.N_hat = Function(FRB)
    c.N_hat0 = Function(FRB)

    # primary variable
    c.u = (c.U_hat, c.g, c.phi_hat)

    dt, kappa = config.params.dt, config.params.kappa
    ADD = inner_product((RB, 0), (RB, 2))
    BDD = inner_product((RB, 0), (RB, 0))
    c.CTD = inner_product((c.CT, 0), (RB, 1))
    c.BTT = inner_product((c.CT, 0), (c.CT, 0))
    ADD.axis = 0
    BDD.axis = 0
    c.CTD.axis = 0
    c.BTT.axis = 0
    c.mat.ABD = inner_product((c.SB, 0), (RB, 2))
    c.mat.BBD = inner_product((c.SB, 0), (RB, 0))
    c.mat.ABD.axis = 0
    c.mat.BBD.axis = 0
    c.la.HelmholtzSolverT = [Helmholtz(ADD, BDD, -np.ones((1, 1, 1)),
                                       (c.K2[0]+2.0/kappa/(c.a[rk]+c.b[rk])/dt)[np.newaxis, :, :]) for rk in range(3)]
    c.TC = [HelmholtzCoeff(config.params.N[0], 1.0, (2./kappa/dt/(c.a[rk]+c.b[rk])-c.K2), c.ST.quad) for rk in range(3)]

    c.hdf5file = RBWriter({'U':c.U[0], 'V':c.U[1], 'W':c.U[2], 'phi':c.phi},
                          chkpoint={'current':{'U':c.U, 'phi':c.phi},
                                    'previous':{}},
                          filename=config.params.solver+'.h5',
                          mesh={'x':c.x0, 'y':c.x1, 'z':c.x2})
    return c

class RBWriter(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        c = config.AttributeDict(context)
        U = c.U_hat.backward(c.U)
        phi = c.phi_hat.backward(c.phi)

def ComputeRHS(rhs, u_hat, g_hat, p_hat, rk, solver, context):
    rhs = KMMRK3_ComputeRHS(rhs, u_hat, g_hat, rk, solver, **context)
    c = context
    w0 = c.work[(rhs[0], 0)]
    w1 = c.work[(rhs[0], 1)]
    diff_T = c.work[(rhs[0], 2)]

    w0 = c.mat.ABD.matvec(p_hat, w0)
    w0 -= c.K2*c.mat.BBD.matvec(p_hat, w1)
    w0[0] -= 0.5*c.BTT[0][0]*(p_hat[-2]+p_hat[-1])*c.K2[0]
    w0[1] -= 0.5*c.BTT[0][1]*(p_hat[-2]-p_hat[-1])*c.K2[0]
    rhs[0] += config.params.dt*(c.a[rk]+c.b[rk])*w0
    c.N_hat = DivRBConvection(c.N_hat, u_hat, g_hat, p_hat, **context)
    #c.N_hat = StandardRBConvection(c.N_hat, u_hat, g_hat, p_hat, **context)
    rhs[2] = -2./params.kappa/(c.a[rk]+c.b[rk])*(c.N_hat*c.a[rk] + c.N_hat0*c.b[rk])
    c.N_hat0[:] = c.N_hat
    diff_T = c.TC[rk].matvec(p_hat, diff_T)
    rhs[2] += diff_T
    return rhs

def solve_linear(u_hat, g_hat, p_hat, rhs, rk, context):
    u_hat, g_hat = KMMRK3_solve_linear(u_hat, g_hat, rhs, rk, **context)
    p_hat = context.la.HelmholtzSolverT[rk](p_hat, rhs[2])
    return u_hat, g_hat, p_hat

def DivRBConvection(rhs, u_hat, g_hat, p_hat,
                    phi0, Ua, mat, Kx, VFSp, FRBp, FSBp, FSTp, work, **context):
    uT_hat = work[(p_hat, 0)]
    F_tmp = work[(u_hat, 0)]

    phi0 = FRBp.backward(p_hat, phi0)
    Ua = VFSp.backward(u_hat, Ua)
    uT_hat = FSBp.forward(phi0*Ua[0], uT_hat)
    F_tmp[0] = mat.CDB.matvec(uT_hat, F_tmp[0])
    uT_hat = FSTp.forward(phi0*Ua[1], uT_hat)
    F_tmp[1] = mat.BDD.matvec(1j*Kx[1]*uT_hat, F_tmp[1])
    uT_hat = FSTp.forward(phi0*Ua[2], uT_hat)
    F_tmp[2] = mat.BDD.matvec(1j*Kx[2]*uT_hat, F_tmp[2])
    rhs[:] = np.sum(F_tmp, axis=0)
    return rhs

def StandardRBConvection(rhs, u_hat, g_hat, p_hat,
                         N_hat, phi0, Ua, mat, Kx, VFSp, FCTp, FSTp, CTD,
                         BTT, work, **context):
    # project to Chebyshev basis. Requires modification due to nonhomogen bc
    dTdx_hat = work[(p_hat, 0)]
    dTdxi = work[(Ua, 0)]
    N_s = work[(p_hat, 1)]
    N = work[(Ua[0], 0)]
    diff_T = work[(p_hat, 2)]
    Ua = VFSp.backward(u_hat, Ua)
    dTdx_hat = CTD.matvec(p_hat, dTdx_hat)
    dTdx_hat[0] += 0.5*BTT[0][0]*(p_hat[-2]-p_hat[-1])
    dTdx_hat = BTT.solve(dTdx_hat)
    dTdxi[0] = FCTp.backward(dTdx_hat, dTdxi[0])
    dTdxi[1] = FSTp.backward(1j*Kx[1]*p_hat, dTdxi[1])
    dTdxi[2] = FSTp.backward(1j*Kx[2]*p_hat, dTdxi[2])
    N[:] = Ua[0]*dTdxi[0] + Ua[1]*dTdxi[1] + Ua[2]*dTdxi[2]
    N_hat = FSTp.forward(N, N_hat)
    rhs = mat.BDD.matvec(N_hat, rhs)
    return rhs

def integrate(u_hat, g_hat, p_hat, rhs, dt, solver, context):
    """Regular implicit solver for KMM_RB channel solver"""
    rhs[:] = 0
    for rk in range(3):
        rhs = solver.ComputeRHS(rhs, u_hat, g_hat, p_hat, rk, solver, context)
        u_hat, g_hat, p_hat = solver.solve_linear(u_hat, g_hat, p_hat, rhs, rk, context)
    return (u_hat, g_hat, p_hat), dt, dt

def getintegrator(rhs, u0, solver, context):
    u_hat, g_hat, p_hat = u0
    def func():
        return solver.integrate(u_hat, g_hat, p_hat, rhs, params.dt, solver, context)
    return func
