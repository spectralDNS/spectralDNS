__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from KMM import *

#get_context_KMM = get_context
def get_context():
    #d = get_context_KMM()
    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad=params.Dquad, threads=params.threads,
                            planner_effort=params.planner_effort["dct"])
    SB = ShenBiharmonicBasis(quad=params.Bquad, threads=params.threads,
                             planner_effort=params.planner_effort["dct"])

    Nu = params.N[0]-2   # Number of velocity modes in Shen basis
    Nb = params.N[0]-4   # Number of velocity modes in Shen biharmonic basis
    u_slice = slice(0, Nu)
    v_slice = slice(0, Nb)

    FST = SlabShen_R2C(params.N, params.L, comm, threads=params.threads,
                       communication=params.communication,
                       planner_effort=params.planner_effort,
                       dealias_cheb=params.dealias_cheb)

    float, complex, mpitype = datatypes("double")

    # Mesh variables
    X = FST.get_local_mesh(ST)
    x0, x1, x2 = FST.get_mesh_dims(ST)
    K = FST.get_scaled_local_wavenumbermesh()
        
    K2 = K[1]*K[1]+K[2]*K[2]
    K_over_K2 = K.astype(float) / np.where(K2==0, 1, K2).astype(float)

    # Solution variables
    U  = zeros((3,)+FST.real_shape(), dtype=float)
    U0 = zeros((3,)+FST.real_shape(), dtype=float)
    U_hat  = zeros((3,)+FST.complex_shape(), dtype=complex)
    U_hat0 = zeros((3,)+FST.complex_shape(), dtype=complex)
    g = zeros(FST.complex_shape(), dtype=complex)
    
    # primary variable
    u = (U_hat, g)

    H_hat  = zeros((3,)+FST.complex_shape(), dtype=complex)
    H_hat0 = zeros((3,)+FST.complex_shape(), dtype=complex)
    H_hat1 = zeros((3,)+FST.complex_shape(), dtype=complex)

    dU = zeros((3,)+FST.complex_shape(), dtype=complex)
    hv = zeros((2,)+FST.complex_shape(), dtype=complex),
    hg = zeros((2,)+FST.complex_shape(), dtype=complex),
    h1 = zeros((2, 2, N[0]), dtype=complex)
    
    Source = zeros((3,)+FST.real_shape(), dtype=float)
    Sk = zeros((3,)+FST.complex_shape(), dtype=complex)

    work = work_arrays()
    
    nu, dt, N = params.nu, params.dt, params.N
    K4 = K2**2
    kx = K[0, :, 0, 0]
    
    # Collect all linear algebra solvers
    # RK 3 requires three solvers because of the three different coefficients
    la = config.AttributeDict(dict(
        HelmholtzSolverG = [Helmholtz(N[0], np.sqrt(K[1, 0]**2+K[2, 0]**2+2.0/nu/(a[rk]+b[rk])/dt),
                                    ST.quad, False) for rk in range(3)],
        BiharmonicSolverU = [Biharmonic(N[0], -nu*(a[rk]+b[rk])*dt/2., 1.+nu*(a[rk]+b[rk])*dt*K2[0],
                                        -(K2[0] + nu*(a[rk]+b[rk])*dt/2.*K4[0]), SB.quad) for rk in range(3)],
        HelmholtzSolverU0 = [Helmholtz(N[0], np.sqrt(2./nu/(a[rk]+b[rk])/dt), ST.quad, False) for rk in range(3)],
        TDMASolverD = TDMA(ST.quad, False)
        )
    )

    alfa = K2[0] - 2.0/nu/dt
    # Collect all matrices
    mat = config.AttributeDict(dict(
        CDD = CDDmat(kx),
        AC = [BiharmonicCoeff(K[0, :, 0, 0], nu*(a[rk]+b[rk])*dt/2., (1. - nu*(a[rk]+b[rk])*dt*K2[0]),
                            -(K2[0] - nu*(a[rk]+b[rk])*dt/2.*K4[0]), SB.quad) for rk in range(3)],
        AB = [HelmholtzCoeff(K[0, :, 0, 0], -1.0, -(K2[0] - 2.0/nu/dt/(a[rk]+b[rk])), ST.quad) for rk in range(3)],

        # Matrices for biharmonic equation
        CBD = CBDmat(kx),
        ABB = ABBmat(kx),
        BBB = BBBmat(kx, SB.quad),
        SBB = SBBmat(kx),
        # Matrices for Helmholtz equation
        ADD = ADDmat(kx),
        BDD = BDDmat(kx, ST.quad),
        BBD = BBDmat(kx, SB.quad),
        CDB = CDBmat(kx)
        )
    )
    
    hdf5file = KMMWriter({"U":U[0], "V":U[1], "W":U[2]},
                         chkpoint={'current':{'U':U}, 'previous':{'U':U0}},
                         filename=params.solver+".h5",
                         mesh={"x": x0, "y": x1, "z": x2})
    
    # RK parameters
    a = (8./15., 5./12., 3./4.)
    b = (0.0, -17./60., -5./12.)
    return config.AttributeDict(locals())


@optimizer
def add_linear(rhs, u, g, work, AB, AC, SBB, ABB, BBB, nu, dt, K2, K4, a, b):
    diff_u = work[(g, 0)]
    diff_g = work[(g, 1)]
    
    # Compute diffusion for g-equation
    diff_g = AB.matvec(g, diff_g)
    
    # Compute diffusion++ for u-equation
    diff_u[:] = nu*(a+b)*dt/2.*SBB.matvec(u)
    diff_u += (1. - nu*(a+b)*dt*K2)*ABB.matvec(u)
    diff_u -= (K2 - nu*(a+b)*dt/2.*K2**2)*BBB.matvec(u)

    rhs[0] += diff_u
    rhs[1] += diff_g
    
    return rhs

def ComputeRHS(rhs, u_hat, g_hat, rk, solver,
               H_hat, H_hat1, H_hat0, FST, ST, SB, work, K, K2, K4, hv,
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
    H_hat = solver.conv(H_hat, u_hat, g_hat, K, FST, SB, ST, work, mat, la)

    hv[1] = -K2*mat.BBD.matvec(H_hat[0])
    #hv[:] = FST.fss(H[0], hv, SB)
    #hv *= -K2
    hv[1] -= 1j*K[1]*mat.CBD.matvec(H_hat[1])
    hv[1] -= 1j*K[2]*mat.CBD.matvec(H_hat[2])    
    hg[1] = 1j*K[1]*mat.BDD.matvec(H_hat[2]) - 1j*K[2]*mat.BDD.matvec(H_hat[1])

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

    f_hat = work[(u_hat[0], 0, False)]
    
    u_hat[0] = la.BiharmonicSolverU[rk](u_hat[0], rhs[0])
    g_hat = la.HelmholtzSolverG[rk](g_hat, rhs[1])

    if rank == 0:
        u0_hat = work[((2, params.N[0]), complex, 0)]
        h0_hat = work[((2, params.N[0]), complex, 1)]
        u0_hat[0] = u_hat[1, :, 0, 0]
        u0_hat[1] = u_hat[2, :, 0, 0]

    # Compute v_hat and w_hat from u_hat and g_hat
    f_hat[:] = -mat.CDB.matvec(u_hat[0])
    f_hat = la.TDMASolverD(f_hat)
    u_hat[1] = -1j*(K_over_K2[1]*f_hat - K_over_K2[2]*g_hat)
    u_hat[2] = -1j*(K_over_K2[2]*f_hat + K_over_K2[1]*g_hat)

    # Remains to fix wavenumber 0    
    if rank == 0:
        w = work[((params.N[0], ), complex, 0, False)]
        
        h0_hat[0] = H_hat[1, :, 0, 0]
        h0_hat[1] = H_hat[2, :, 0, 0]
        
        h1[1, 0] = mat.BDD.matvec(h0_hat[0])
        h1[1, 1] = mat.BDD.matvec(h0_hat[1])
        h1[1, 0] -= Sk[1, :, 0, 0]  # Subtract constant pressure gradient
        
        beta = 2./params.nu/(a[rk]+b[rk])/params.dt
        w[:] = beta*(a[rk]*h1[1, 0] + b[rk]*h1[0, 0])*params.dt
        w -= mat.ADD.matvec(u0_hat[0])
        w += beta*mat.BDD.matvec(u0_hat[0])    
        u0_hat[0] = la.HelmholtzSolverU0[rk](u0_hat[0], w)
        
        w[:] = beta*(a[rk]*h1[1, 1] + b[rk]*h1[0, 1])*params.dt
        w -= mat.ADD.matvec(u0_hat[1])
        w += beta*mat.BDD.matvec(u0_hat[1])    
        u0_hat[1] = la.HelmholtzSolverU0[rk](u0_hat[1], w)
        
        h1[0]  = h1[1]
        
        u_hat[1, :, 0, 0] = u0_hat[0]
        u_hat[2, :, 0, 0] = u0_hat[1]
    
    return u_hat, g_hat

def integrate(u_hat, g_hat, rhs, dt, solver, context):
    """Three stage Runge Kutta integrator for KMM channel solver"""
    rhs[:] = 0
    context.hv[0] = 0
    context.hg[0] = 0
    context.h1[:] = 0    
    for rk in range(3):            
        rhs = solver.ComputeRHS(rhs, u_hat, g_hat, rk, solver, **context)
        u_hat, g_hat = solver.solve_linear(u_hat, g_hat, rhs, rk, **context)
    
    return (u_hat, g_hat), dt, dt
