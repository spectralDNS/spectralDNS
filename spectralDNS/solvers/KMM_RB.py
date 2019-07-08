__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2018-08-29"
__copyright__ = "Copyright (C) 2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=function-redefined,unbalanced-tuple-unpacking,unused-variable,unused-argument

from .KMM import *
from .spectralinit import end_of_tstep

KMM_context = get_context
KMM_ComputeRHS = ComputeRHS
KMM_solve_linear = solve_linear

def get_context():
    c = KMM_context()

    c.RB = RB = Basis(config.params.N[0], 'C', bc=(0, 1))
    c.FRB = FRB = TensorProductSpace(comm, (RB, c.K0, c.K1), **c.kw0)
    c.FRBp = FRBp = TensorProductSpace(comm, (RB, c.K0p, c.K1p), **c.kw0)

    c.dU = Function(c.VFS)  # rhs vector for integrator. Now three components, not two
    c.phi = Array(FRB)
    c.phi0 = Array(FRBp)
    c.phi_hat = Function(FRB)
    c.phi_hat0 = Function(FRB)
    c.phi_hat1 = Function(FRB)
    c.phi_ab = Function(FRB)
    c.UCN = Array(c.VFSp)
    c.N_hat = Function(FRB)
    c.N_hat0 = Function(FRB)

    # primary variable
    c.u = (c.U_hat, c.g, c.phi_hat)

    dt, kappa = config.params.dt, config.params.kappa
    ADD = inner_product((RB, 0), (RB, 2))
    BDD = inner_product((RB, 0), (RB, 0))
    c.CTD = inner_product((c.CT, 0), (RB, 1))
    c.BTT = inner_product((c.CT, 0), (c.CT, 0))
    c.mat.ABD = inner_product((c.SB, 0), (RB, 2))
    c.mat.BBD = inner_product((c.SB, 0), (RB, 0))
    c.la.HelmholtzSolverT = Helmholtz(ADD, BDD,
                                      -np.ones((1, 1, 1)),
                                      (c.K2[0]+2.0/kappa/dt)[np.newaxis, :, :])
    c.TC = HelmholtzCoeff(config.params.N[0], 1.0, (2./kappa/dt-c.K2), 0)

    c.hdf5file = RBFile(config.params.solver,
                        checkpoint={'space': c.VFS,
                                    'data': {'0': {'U': [c.U_hat], 'phi': [c.phi_hat]},
                                             '1': {'U': [c.U_hat0], 'phi': [c.phi_hat0]}}},
                        results={'space': c.VFS,
                                 'data': {'U': [c.U], 'phi': [c.phi]}})

    return c

class RBFile(HDF5File):
    def update_components(self, U, U_hat, phi, phi_hat, **context):
        """Transform to real data when storing the solution"""
        U = U_hat.backward(U)
        phi = phi_hat.backward(phi)

def end_of_tstep(context):
    context.phi_hat1[:] = context.phi_hat0
    context.phi_hat0[:] = context.phi_hat
    context.U_hat0[:] = context.U_hat
    context.H_hat1[:] = context.H_hat
    context.N_hat0[:] = context.N_hat
    return False

def ComputeRHS(rhs, u_hat, g_hat, p_hat, solver, context):
    rhs = KMM_ComputeRHS(rhs, u_hat, g_hat, solver, **context)
    c = context
    w0 = c.work[(rhs[0], 0, True)]
    w1 = c.work[(rhs[0], 1, True)]
    diff_T = c.work[(rhs[0], 2, True)]
    phi_ab = c.phi_ab
    phi_ab[:] = 1.5*c.phi_hat0 - 0.5*c.phi_hat1
    w0 = c.mat.ABD.matvec(phi_ab, w0)
    w0 -= c.K2*c.mat.BBD.matvec(phi_ab, w1)
    w0[0] -= 0.5*c.BTT[0][0]*(phi_ab[-2]+phi_ab[-1])*c.K2[0]
    w0[1] -= 0.5*c.BTT[0][1]*(phi_ab[-2]-phi_ab[-1])*c.K2[0]
    rhs[0] += config.params.dt*w0
    rhs[2] = DivRBConvection(rhs[2], u_hat, g_hat, p_hat, **context)
    #rhs[2] = DivABConvection(rhs[2], u_hat, g_hat, p_hat, **context)
    #rhs[2] = StandardRBConvection(rhs[2], u_hat, g_hat, p_hat, **context)

    rhs[2] *= -2./params.kappa
    diff_T = c.TC.matvec(c.phi_hat0, diff_T)
    rhs[2] += diff_T
    return rhs

def solve_linear(u_hat, g_hat, p_hat, rhs, context):
    u_hat, g_hat = KMM_solve_linear(u_hat, g_hat, rhs, **context)
    p_hat = context.la.HelmholtzSolverT(p_hat, rhs[2])
    return u_hat, g_hat, p_hat

def DivRBConvection(rhs, u_hat, g_hat, p_hat,
                    UCN, VFSp, U_hat0, phi_ab, phi0, mat, Kx, N_hat,
                    phi_hat0, phi_hat1, FRBp, FSBp, FSTp, work, **context):
    uT_hat = work[(p_hat, 0, True)]
    F_tmp = work[(u_hat, 0, True)]

    UCN = VFSp.backward(0.5*(u_hat + U_hat0), UCN)
    phi_ab[:] = 1.5*phi_hat0 - 0.5*phi_hat1
    phi0 = FRBp.backward(phi_ab, phi0)
    uT_hat = FSBp.forward(phi0*UCN[0], uT_hat)
    F_tmp[0] = mat.CDB.matvec(uT_hat, F_tmp[0])
    uT_hat = FSTp.forward(phi0*UCN[1], uT_hat)
    F_tmp[1] = mat.BDD.matvec(1j*Kx[1]*uT_hat, F_tmp[1])
    uT_hat = FSTp.forward(phi0*UCN[2], uT_hat)
    F_tmp[2] = mat.BDD.matvec(1j*Kx[2]*uT_hat, F_tmp[2])
    N_hat[:] = np.sum(F_tmp, axis=0)
    rhs[:] = N_hat
    return rhs

def DivABConvection(rhs, u_hat, g_hat, p_hat,
                    UCN, VFSp, U_hat0, phi_ab, phi0, mat, Kx, N_hat, N_hat0,
                    phi_hat0, phi_hat1, FRBp, FSBp, FSTp, work, **context):
    uT_hat = work[(p_hat, 0, True)]
    F_tmp = work[(u_hat, 0, True)]

    UCN = VFSp.backward(u_hat, UCN)
    phi0 = FRBp.backward(p_hat, phi0)
    uT_hat = FSBp.forward(phi0*UCN[0], uT_hat)
    F_tmp[0] = mat.CDB.matvec(uT_hat, F_tmp[0])
    uT_hat = FSTp.forward(phi0*UCN[1], uT_hat)
    F_tmp[1] = mat.BDD.matvec(1j*Kx[1]*uT_hat, F_tmp[1])
    uT_hat = FSTp.forward(phi0*UCN[2], uT_hat)
    F_tmp[2] = mat.BDD.matvec(1j*Kx[2]*uT_hat, F_tmp[2])
    N_hat[:] = np.sum(F_tmp, axis=0)
    rhs[:] = 1.5*N_hat - 0.5*N_hat0
    return rhs

def StandardRBConvection(rhs, u_hat, g_hat, p_hat,
                         UCN, VFSp, U_hat0, mat, Kx, N_hat, N_hat0,
                         FCTp, FSTp, work, CTD, BTT, **context):
    # project to Chebyshev basis. Requires modification due to nonhomogen bc
    dTdx_hat = work[(p_hat, 0, True)]
    dTdxi = work[(UCN, 0, True)]
    N_s = work[(p_hat, 1, True)]
    N = work[(UCN[0], 0, True)]
    diff_T = work[(p_hat, 2, True)]
    UCN = VFSp.backward(U_hat0, UCN)
    dTdx_hat = CTD.matvec(p_hat, dTdx_hat)
    dTdx_hat[0] += 0.5*BTT[0][0]*(p_hat[-2]-p_hat[-1])
    dTdx_hat = BTT.solve(dTdx_hat)
    dTdxi[0] = FCTp.backward(dTdx_hat, dTdxi[0])
    dTdxi[1] = FSTp.backward(1j*Kx[1]*p_hat, dTdxi[1])
    dTdxi[2] = FSTp.backward(1j*Kx[2]*p_hat, dTdxi[2])
    N[:] = UCN[0]*dTdxi[0] + UCN[1]*dTdxi[1] + UCN[2]*dTdxi[2]
    N_hat = FSTp.forward(N, N_hat)
    rhs = mat.BDD.matvec(1.5*N_hat-0.5*N_hat0, rhs)
    return rhs

def integrate(u_hat, g_hat, p_hat, rhs, dt, solver, context):
    """Regular implicit solver for KMM_RB channel solver"""
    rhs[:] = 0
    rhs = solver.ComputeRHS(rhs, u_hat, g_hat, p_hat, solver, context)
    if context.mask is not None:
        rhs *= context.mask
    u_hat, g_hat, p_hat = solver.solve_linear(u_hat, g_hat, p_hat, rhs, context)
    return (u_hat, g_hat, p_hat), dt, dt

def getintegrator(rhs, u0, solver, context):
    u_hat, g_hat, p_hat = u0
    def func():
        return solver.integrate(u_hat, g_hat, p_hat, rhs, params.dt, solver, context)
    return func
