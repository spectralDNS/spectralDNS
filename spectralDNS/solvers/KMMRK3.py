__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .KMM import *
from .spectralinit import end_of_tstep

def get_context():
    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(params.N[0], quad=params.Dquad)
    SB = ShenBiharmonicBasis(params.N[0], quad=params.Bquad)
    CT = Basis(params.N[0], quad=params.Dquad)
    ST0 = ShenDirichletBasis(params.N[0], quad=params.Dquad, plan=True) # For 1D problem
    K0 = C2CBasis(params.N[1], domain=(0, params.L[1]))
    K1 = R2CBasis(params.N[2], domain=(0, params.L[2]))

    #CT = ST.CT  # Chebyshev transform
    FST = TensorProductSpace(comm, (ST, K0, K1), **{'threads':params.threads, 'planner_effort':params.planner_effort["dct"]})    # Dirichlet
    FSB = TensorProductSpace(comm, (SB, K0, K1), **{'threads':params.threads, 'planner_effort':params.planner_effort["dct"]})    # Biharmonic
    FCT = TensorProductSpace(comm, (CT, K0, K1), **{'threads':params.threads, 'planner_effort':params.planner_effort["dct"]})    # Regular Chebyshev
    VFS = VectorTensorProductSpace([FSB, FST, FST])

    # Padded
    kw = {'padding_factor': 1.5 if params.dealias == '3/2-rule' else 1,
          'dealias_direct': params.dealias == '2/3-rule'}
    if params.dealias == '3/2-rule':
        # Requires new bases due to planning and transforms on different size arrays
        STp = ShenDirichletBasis(params.N[0], quad=params.Dquad)
        SBp = ShenBiharmonicBasis(params.N[0], quad=params.Bquad)
        CTp = Basis(params.N[0], quad=params.Dquad)
    else:
        STp, SBp, CTp = ST, SB, CT

    K0p = C2CBasis(params.N[1], domain=(0, params.L[1]), **kw)
    K1p = R2CBasis(params.N[2], domain=(0, params.L[2]), **kw)
    FSTp = TensorProductSpace(comm, (STp, K0p, K1p), **{'threads':params.threads, 'planner_effort':params.planner_effort["dct"]})
    FSBp = TensorProductSpace(comm, (SBp, K0p, K1p), **{'threads':params.threads, 'planner_effort':params.planner_effort["dct"]})
    FCTp = TensorProductSpace(comm, (CTp, K0p, K1p), **{'threads':params.threads, 'planner_effort':params.planner_effort["dct"]})
    VFSp = VectorTensorProductSpace([FSBp, FSTp, FSTp])

    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nb = params.N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)

    float, complex, mpitype = datatypes("double")

    # Mesh variables
    X = FST.local_mesh(True)
    x0, x1, x2 = FST.mesh()
    K = FST.local_wavenumbers(scaled=True)

    # Solution variables
    U = Array(VFS, False)
    U_hat = Array(VFS)
    g = Array(FST)

    # primary variable
    u = (U_hat, g)

    nu, dt, N = params.nu, params.dt, params.N

    H_hat = Array(VFS)

    dU = Array(VFS)
    hv = zeros((2,)+FST.local_shape(), dtype=complex)
    hg = zeros((2,)+FST.local_shape(), dtype=complex)
    h1 = zeros((2, 2, N[0]), dtype=complex)

    Source = Array(VFS, False)
    Sk = Array(VFS)

    K2 = K[1]*K[1]+K[2]*K[2]
    K4 = K2**2
    kx = K[0][:, 0, 0]

    # Set Nyquist frequency to zero on K that is used for odd derivatives
    K = FST.local_wavenumbers(scaled=True, eliminate_highest_freq=True)
    K_over_K2 = np.zeros((2,)+g.shape)
    for i in range(2):
        K_over_K2[i] = K[i+1] / np.where(K2==0, 1, K2)

    work = work_arrays()

    # RK parameters
    a = (8./15., 5./12., 3./4.)
    b = (0.0, -17./60., -5./12.)

    alfa = K2[0] - 2.0/nu/dt
    # Collect all matrices
    mat = config.AttributeDict(dict(
        CDD = inner_product((ST, 0), (ST, 1)),
        AC = [BiharmonicCoeff(N[0], nu*(a[rk]+b[rk])*dt/2., (1. - nu*(a[rk]+b[rk])*dt*K2[0]),
                            -(K2[0] - nu*(a[rk]+b[rk])*dt/2.*K4[0]), SB.quad) for rk in range(3)],
        AB = [HelmholtzCoeff(N[0], 1.0, -(K2[0] - 2.0/nu/dt/(a[rk]+b[rk])), ST.quad) for rk in range(3)],

        # Matrices for biharmonic equation
        CBD = inner_product((SB, 0), (ST, 1)),
        ABB = inner_product((SB, 0), (SB, 2)),
        BBB = inner_product((SB, 0), (SB, 0)),
        SBB = inner_product((SB, 0), (SB, 4)),
        # Matrices for Helmholtz equation
        ADD = inner_product((ST, 0), (ST, 2)),
        BDD = inner_product((ST, 0), (ST, 0)),
        BBD = inner_product((SB, 0), (ST, 0)),
        CDB = inner_product((ST, 0), (SB, 1)),
        ADD0 = inner_product((ST0, 0), (ST0, 2)),
        BDD0 = inner_product((ST0, 0), (ST0, 0)),
        )
    )
    mat.ADD.axis = 0
    mat.BDD.axis = 0
    mat.SBB.axis = 0

    # Collect all linear algebra solvers
    # RK 3 requires three solvers because of the three different coefficients
    rk = 0
    la = config.AttributeDict(dict(
        HelmholtzSolverG = [Helmholtz(mat.ADD, mat.BDD, -np.ones((1,1,1)),
                            (K2[0]+2.0/nu/(a[rk]+b[rk])/dt)[np.newaxis, :, :])
                            for rk in range(3)],
        BiharmonicSolverU = [Biharmonic(mat.SBB, mat.ABB, mat.BBB, -nu*(a[rk]+b[rk])*dt/2.*np.ones((1,1,1)),
                                        (1.+nu*(a[rk]+b[rk])*dt*K2[0])[np.newaxis,:,:],
                                        -(K2[0] + nu*(a[rk]+b[rk])*dt/2.*K4[0])[np.newaxis,:,:])
                             for rk in range(3)],
        HelmholtzSolverU0 = [old_Helmholtz(N[0], np.sqrt(2./nu/(a[rk]+b[rk])/dt), ST) for rk in range(3)],
        TDMASolverD = TDMA(inner_product((ST, 0), (ST, 0)))
        )
    )

    del rk

    hdf5file = KMMRK3Writer({"U":U[0], "V":U[1], "W":U[2]},
                             chkpoint={'current':{'U':U}, 'previous':{}},
                             filename=params.solver+".h5",
                             mesh={"x": x0, "y": x1, "z": x2})

    return config.AttributeDict(locals())

class KMMRK3Writer(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        U = get_velocity(**context)    # updates U from U_hat

@optimizer
def add_linear(rhs, u, g, work, AB, AC, SBB, ABB, BBB, nu, dt, K2, K4, a, b):
    diff_u = work[(g, 0)]
    diff_g = work[(g, 1, False)]
    w0 = work[(g, 2, False)]

    # Compute diffusion for g-equation
    diff_g = AB.matvec(g, diff_g)

    # Compute diffusion++ for u-equation
    diff_u[:] = nu*(a+b)*dt/2.*SBB.matvec(u, w0)
    diff_u += (1. - nu*(a+b)*dt*K2)*ABB.matvec(u, w0)
    diff_u -= (K2 - nu*(a+b)*dt/2.*K2**2)*BBB.matvec(u, w0)

    rhs[0] += diff_u
    rhs[1] += diff_g

    return rhs

def ComputeRHS(rhs, u_hat, g_hat, rk, solver,
               H_hat, VFSp, FSTp, FSBp, FCTp, work, K, K2, K4, hv,
               hg, a, b, K_over_K2, la, mat, **context):

    """Compute right hand side of Navier Stokes

    args:
        rhs         The right hand side to be returned
        u_hat       The FST of the velocity at current time.
        g_hat       The FST of the curl in wall normal direction
        rk          The step in the Runge Kutta integrator
        solver      The current solver module

    Remaining args are extracted from context

    """

    # Nonlinear convection term at current u_hat
    H_hat = solver.conv(H_hat, u_hat, g_hat, K, VFSp, FSTp, FSBp, FCTp, work, mat, la)

    w0 = work[(H_hat[0], 0, False)]
    w1 = work[(H_hat[0], 1, False)]
    hv[1] = -K2*mat.BBD.matvec(H_hat[0], w0)
    #hv[:] = FST.scalar_product(H[0], hv, SB)
    #hv *= -K2
    hv[1] -= 1j*K[1]*mat.CBD.matvec(H_hat[1], w0)
    hv[1] -= 1j*K[2]*mat.CBD.matvec(H_hat[2], w0)
    hg[1] = 1j*K[1]*mat.BDD.matvec(H_hat[2], w0) - 1j*K[2]*mat.BDD.matvec(H_hat[1], w1)

    rhs[0] = (hv[1]*a[rk] + hv[0]*b[rk])*params.dt
    rhs[1] = (hg[1]*a[rk] + hg[0]*b[rk])*2./params.nu/(a[rk]+b[rk])

    hv[0] = hv[1]
    hg[0] = hg[1]

    rhs = solver.add_linear(rhs, u_hat[0], g_hat, work, mat.AB[rk], mat.AC[rk],
                            mat.SBB, mat.ABB, mat.BBB, params.nu, params.dt,
                            K2, K4, a[rk], b[rk])
    return rhs

def solve_linear(u_hat, g_hat, rhs, rk,
                 work, la, mat, H_hat, Sk, h1, a, b, K_over_K2, **context):

    f_hat = work[(u_hat[0], 0)]
    w0 = work[(u_hat[0], 1, False)]

    u_hat[0] = la.BiharmonicSolverU[rk](u_hat[0], rhs[0])
    g_hat = la.HelmholtzSolverG[rk](g_hat, rhs[1])

    if rank == 0:
        u0_hat = work[((2, params.N[0]), complex, 0)]
        h0_hat = work[((2, params.N[0]), complex, 1)]
        u0_hat[0] = u_hat[1, :, 0, 0]
        u0_hat[1] = u_hat[2, :, 0, 0]

    # Compute v_hat and w_hat from u_hat and g_hat
    f_hat -= mat.CDB.matvec(u_hat[0], w0)
    f_hat = la.TDMASolverD(f_hat)
    u_hat = compute_vw(u_hat, f_hat, g_hat, K_over_K2)

    # Remains to fix wavenumber 0
    if rank == 0:
        w = work[((params.N[0], ), complex, 0)]
        w1 = work[((params.N[0], ), complex, 1, False)]

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

    return u_hat, g_hat

def integrate(u_hat, g_hat, rhs, dt, solver, context):
    """Three stage Runge Kutta integrator for KMM channel solver"""
    for rk in range(3):
        rhs = solver.ComputeRHS(rhs, u_hat, g_hat, rk, solver, **context)
        u_hat, g_hat = solver.solve_linear(u_hat, g_hat, rhs, rk, **context)

    return (u_hat, g_hat), dt, dt
