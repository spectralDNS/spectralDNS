__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=function-redefined,unbalanced-tuple-unpacking,unused-variable,unused-argument

from .KMM import *
from .spectralinit import end_of_tstep

KMM_context = get_context

def get_context():
    c = KMM_context()
    del c.U0, c.U_hat0

    nu, dt, N = params.nu, params.dt, params.N
    del c.H_hat0, c.H_hat1
    c.hv = np.zeros((2,)+c.FST.local_shape(), dtype=complex)
    c.hg = np.zeros((2,)+c.FST.local_shape(), dtype=complex)
    c.h1 = np.zeros((2, 2, N[0]), dtype=complex)

    # RK parameters
    c.a = a = (8./15., 5./12., 3./4.)
    c.b = b = (0.0, -17./60., -5./12.)

    # Collect all matrices
    c.mat.AC = [BiharmonicCoeff(N[0], nu*(a[rk]+b[rk])*dt/2., (1. - nu*(a[rk]+b[rk])*dt*c.K2),
                                -(c.K2 - nu*(a[rk]+b[rk])*dt/2.*c.K4), c.SB.quad) for rk in range(3)]
    c.mat.AB = [HelmholtzCoeff(N[0], 1.0, -(c.K2 - 2.0/nu/dt/(a[rk]+b[rk])), c.ST.quad) for rk in range(3)]

    # Collect all linear algebra solvers
    # RK 3 requires three solvers because of the three different coefficients
    c.la = config.AttributeDict(
        dict(HelmholtzSolverG=[Helmholtz(c.mat.ADD, c.mat.BDD, -np.ones((1, 1, 1)),
                                         (c.K2+2.0/nu/(a[rk]+b[rk])/dt))
                               for rk in range(3)],
             BiharmonicSolverU=[Biharmonic(c.mat.SBB, c.mat.ABB, c.mat.BBB, -nu*(a[rk]+b[rk])*dt/2.*np.ones((1, 1, 1)),
                                           (1.+nu*(a[rk]+b[rk])*dt*c.K2),
                                           -(c.K2 + nu*(a[rk]+b[rk])*dt/2.*c.K4))
                                for rk in range(3)],
             HelmholtzSolverU0=[Helmholtz(c.mat.ADD0, c.mat.BDD0, np.array([-1]), np.array([2./nu/(a[rk]+b[rk])/dt])) for rk in range(3)],
             TDMASolverD=TDMA(inner_product((c.ST, 0), (c.ST, 0)))))

    del c.hdf5file.checkpoint['data']['1'] # No need for previous time step

    return c

@optimizer
def add_linear(rhs, u, g, work, AB, AC, SBB, ABB, BBB, nu, dt, K2, K4, a, b):
    diff_u = work[(g, 0, False)]
    diff_g = work[(g, 1, False)]
    w0 = work[(g, 2, False)]

    # Compute diffusion for g-equation
    diff_g = AB.matvec(g, diff_g)

    # Compute diffusion++ for u-equation
    diff_u[:] = nu*(a+b)*dt/2.*SBB.matvec(u, w0)
    diff_u += (1. - nu*(a+b)*dt*K2)*ABB.matvec(u, w0)
    diff_u -= (K2 - nu*(a+b)*dt/2.*K4)*BBB.matvec(u, w0)

    rhs[0] += diff_u
    rhs[1] += diff_g

    return rhs

def ComputeRHS(rhs, u_hat, g_hat, rk, solver,
               H_hat, VFSp, FSTp, FSBp, FCTp, work, Kx, K2, K4, hv,
               hg, a, b, la, mat, u_dealias, **context):

    """Compute right hand side of Navier Stokes

    Parameters
    ----------
        rhs : array
            The right hand side to be returned
        u_hat : array
            The FST of the velocity at current time.
        g_hat : array
            The FST of the curl in wall normal direction
        rk : int
            The step in the Runge Kutta integrator
        solver : module
            The current solver module

    Remaining args are extracted from context

    """

    # Nonlinear convection term at current u_hat
    H_hat = solver.conv(H_hat, u_hat, g_hat, Kx, VFSp, FSTp, FSBp, FCTp, work, mat, la, u_dealias)

    w0 = work[(H_hat[0], 0, False)]
    w1 = work[(H_hat[0], 1, False)]
    hv[1] = -K2*mat.BBD.matvec(H_hat[0], w0)
    #hv[:] = FST.scalar_product(H[0], hv, SB)
    #hv *= -K2
    hv[1] -= 1j*Kx[1]*mat.CBD.matvec(H_hat[1], w0)
    hv[1] -= 1j*Kx[2]*mat.CBD.matvec(H_hat[2], w0)
    hg[1] = 1j*Kx[1]*mat.BDD.matvec(H_hat[2], w0) - 1j*Kx[2]*mat.BDD.matvec(H_hat[1], w1)

    rhs[0] = (hv[1]*a[rk] + hv[0]*b[rk])*params.dt
    rhs[1] = (hg[1]*a[rk] + hg[0]*b[rk])*2./params.nu/(a[rk]+b[rk])

    hv[0] = hv[1]
    hg[0] = hg[1]

    rhs = solver.add_linear(rhs, u_hat[0], g_hat, work, mat.AB[rk], mat.AC[rk],
                            mat.SBB, mat.ABB, mat.BBB, params.nu, params.dt,
                            K2, K4, a[rk], b[rk])
    return rhs

def solve_linear(u_hat, g_hat, rhs, rk,
                 work, la, mat, H_hat, Sk, h1, a, b, K_over_K2, u0_hat, h0_hat,
                 w, w1, **context):

    f_hat = work[(u_hat[0], 0, True)]
    w0 = work[(u_hat[0], 1, False)]

    u_hat[0] = la.BiharmonicSolverU[rk](u_hat[0], rhs[0])
    g_hat = la.HelmholtzSolverG[rk](g_hat, rhs[1])

    if rank == 0:
        #u0_hat = work[((2, params.N[0]), complex, 0)]
        #h0_hat = work[((2, params.N[0]), complex, 1)]
        u0_hat[0] = u_hat[1, :, 0, 0]
        u0_hat[1] = u_hat[2, :, 0, 0]

    # Compute v_hat and w_hat from u_hat and g_hat
    f_hat -= mat.CDB.matvec(u_hat[0], w0)
    f_hat = la.TDMASolverD(f_hat)
    u_hat = compute_vw(u_hat, f_hat, g_hat, K_over_K2)

    # Remains to fix wavenumber 0
    if rank == 0:
        #w = work[((params.N[0], ), complex, 0)]
        #w1 = work[((params.N[0], ), complex, 1, False)]

        h0_hat[0] = H_hat[1, :, 0, 0]
        h0_hat[1] = H_hat[2, :, 0, 0]

        h1[1, 0] = mat.BDD0.matvec(h0_hat[0], h1[1, 0])
        h1[1, 1] = mat.BDD0.matvec(h0_hat[1], h1[1, 1])
        h1[1, 0] -= Sk[1, :, 0, 0]  # Subtract constant pressure gradient

        beta = 2./params.nu/(a[rk]+b[rk])
        w[:] = beta*(a[rk]*h1[1, 0] + b[rk]*h1[0, 0])
        w += mat.ADD0.matvec(u0_hat[0], w1)
        w += beta/params.dt*mat.BDD0.matvec(u0_hat[0], w1)
        u0_hat[0] = la.HelmholtzSolverU0[rk](u0_hat[0], w)

        w[:] = beta*(a[rk]*h1[1, 1] + b[rk]*h1[0, 1])
        w += mat.ADD0.matvec(u0_hat[1], w1)
        w += beta/params.dt*mat.BDD0.matvec(u0_hat[1], w1)
        u0_hat[1] = la.HelmholtzSolverU0[rk](u0_hat[1], w)

        h1[0] = h1[1]

        u_hat[1, :, 0, 0] = u0_hat[0]
        u_hat[2, :, 0, 0] = u0_hat[1]
        u_hat[0, :, 0, 0] = 0

    return u_hat, g_hat

def integrate(u_hat, g_hat, rhs, dt, solver, context):
    """Three stage Runge Kutta integrator for KMM channel solver"""
    for rk in range(3):
        rhs = solver.ComputeRHS(rhs, u_hat, g_hat, rk, solver, **context)
        u_hat, g_hat = solver.solve_linear(u_hat, g_hat, rhs, rk, **context)

    return (u_hat, g_hat), dt, dt
