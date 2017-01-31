__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-10-29"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from .spectralinit import *
from shenfun.chebyshev.matrices import CDNmat, CDDmat, BDNmat, BDDmat, BDTmat, \
    CNDmat, BNNmat
from ..shen import LUsolve
from ..shen.Matrices import HelmholtzCoeff
from ..shen.la import Helmholtz
from ..shen.shentransform import SlabShen_R2C
from shenfun.chebyshev.bases import ShenDirichletBasis, ShenNeumannBasis, \
    ShenBiharmonicBasis
from shenfun import inner_product
from shenfun.la import TDMA

from functools import wraps

def get_context():
    """Set up context for solver"""

    # Get points and weights for Chebyshev weighted integrals
    ST = ShenDirichletBasis(quad=params.Dquad, threads=params.threads,
                            planner_effort=params.planner_effort["dct"])
    SN = ShenNeumannBasis(quad=params.Nquad, threads=params.threads,
                          planner_effort=params.planner_effort["dct"])
    CT = ST.CT

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
        TDMASolverD = TDMA(inner_product((ST, 0), (ST, 0), N[0])),
        TDMASolverN = TDMA(inner_product((SN, 0), (SN, 0), N[0]))
        )
    )

    alfa = K[1, 0]**2+K[2, 0]**2-2.0/nu/dt

    # Collect all matrices
    mat = config.AttributeDict(dict(
        CDN = inner_product((ST, 0), (SN, 1), N[0]),
        CND = inner_product((SN, 0), (ST, 1), N[0]),
        BDN = inner_product((ST, 0), (SN, 0), N[0]),
        CDD = inner_product((ST, 0), (ST, 1), N[0]),
        BDD = inner_product((ST, 0), (ST, 0), N[0]),
        BDT = inner_product((ST, 0), (CT, 0), N[0]),
        AB = HelmholtzCoeff(K[0, :, 0, 0], -1.0, -alfa, ST.quad)
        )
    )

    hdf5file = IPCSWriter({"U":U[0], "V":U[1], "W":U[2], "P":P},
                          chkpoint={'current':{'U':U, 'P':P}, 'previous':{'U':U0}},
                          filename=params.solver+".h5",
                          mesh={"x": x0, "xp": FST.get_mesh_dim(SN, 0), "y": x1, "z": x2})


    return config.AttributeDict(locals())


class IPCSWriter(HDF5Writer):
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        U = get_velocity(**context)
        P = get_pressure(**context)
        if params.tstep % params.checkpoint == 0:
            c = config.AttributeDict(context)
            U0 = get_velocity(c.U0, c.U_hat0, c.FST, c.ST)

assert params.precision == "double"

def get_pressure(P, P_hat, FST, SN, **context):
    """Compute pressure from context"""
    P = FST.backward(P_hat, P, SN)
    return P

def set_pressure(P_hat, P, FST, SN, **context):
    """Compute pressure from context"""
    P_hat = FST.forward(P, P_hat, SN)
    return P_hat

def get_velocity(U, U_hat, FST, ST, **context):
    for i in range(3):
        U[i] = FST.backward(U_hat[i], U[i], ST)
    return U

def set_velocity(U_hat, U, FST, ST, **context):
    for i in range(3):
        U_hat[i] = FST.forward(U[i], U_hat[i], ST)
    return U_hat

def get_convection(H_hat, U_hat, K, FST, ST, work, mat, la, **context):
    """Compute convection from context"""
    conv = getConvection(params.convection)
    H_hat = conv(H_hat, U_hat, K, FST, ST, work, mat, la)
    return H_hat

def get_pressure(P_hat, P, FST, SN, **context):
    """Get pressure from context"""
    P = FST.backward(P_hat, P, SN)
    return P

def compute_pressure(P_hat, H_hat, U_hat, U_hat0, K, FST, ST, work, mat, la,
                     u_slice, p_slice, **context):
    """Solve for pressure

    Assuming U_hat and U_hat0 are the solutions at two subsequent time steps
    k and k+1, computes the pressure at k+1/2

    """
    conv = getConvection(params.convection)
    H_hat = conv(H_hat, 0.5*(U_hat+U_hat0), K, FST, ST, work, mat, la)
    for i in range(3):
        H_hat[i] = la.TDMASolverD(H_hat[i])
    H_hat *= -1

    dP = work[(P_hat, 0)]
    LUsolve.Mult_Div_3D(params.N[0], K[1, 0], K[2, 0], H_hat[0, u_slice],
                        H_hat[1, u_slice], H_hat[2, u_slice], dP[p_slice])
    P_hat = la.HelmholtzSolverP(P_hat, dP)
    return P_hat

def pressuregrad(rhs, p_hat, mat, work, K, Nu):
    w0 = work[(p_hat, 0, False)]
    # Pressure gradient x-direction
    rhs[0] -= mat.CDN.matvec(p_hat, w0)

    # pressure gradient y-direction
    dP = work[(p_hat, 0, False)]
    dP = mat.BDN.matvec(p_hat, dP)
    rhs[1, :Nu] -= 1j*K[1, :Nu]*dP[:Nu]

    # pressure gradient z-direction
    rhs[2, :Nu] -= 1j*K[2, :Nu]*dP[:Nu]

    return rhs

def pressurerhs(dP, u_hat, K, u_slice, p_slice):
    dP[:] = 0.
    LUsolve.Mult_Div_3D(params.N[0], K[1, 0], K[2, 0], u_hat[0, u_slice],
                        u_hat[1, u_slice], u_hat[2, u_slice], dP[p_slice])

    dP[p_slice] *= -1./params.dt
    return dP

def body_force(rhs, Sk, Nu):
    rhs[0, :Nu] -= Sk[0, :Nu]
    rhs[1, :Nu] -= Sk[1, :Nu]
    rhs[2, :Nu] -= Sk[2, :Nu]
    return rhs

def Cross(c, a, b, FST, S, work):
    Uc = work[(a, 2)]
    Uc = cross1(Uc, a, b)
    c[0] = FST.scalar_product(Uc[0], c[0], S, dealias=params.dealias)
    c[1] = FST.scalar_product(Uc[1], c[1], S, dealias=params.dealias)
    c[2] = FST.scalar_product(Uc[2], c[2], S, dealias=params.dealias)
    return c

def compute_curl(c, u_hat, K, FST, ST, work):
    F_tmp = work[(u_hat, 0)]
    Uc = work[(c, 2)]
    LUsolve.Mult_CTD_3D_n(params.N[0], u_hat[1], u_hat[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FST.backward(F_tmp[1], Uc[1], ST.CT, dealias=params.dealias)
    dwdx = Uc[2] = FST.backward(F_tmp[2], Uc[2], ST.CT, dealias=params.dealias)
    c[0] = FST.backward((1j*K[1]*u_hat[2] - 1j*K[2]*u_hat[1]), c[0], ST, dealias=params.dealias)
    c[1] = FST.backward(1j*K[2]*u_hat[0], c[1], ST, dealias=params.dealias)
    c[1] -= dwdx
    c[2] = FST.backward(1j*K[1]*u_hat[0], c[2], ST, dealias=params.dealias)
    c[2] *= -1.0
    c[2] += dvdx
    return c

def Div(a_hat):
    Uc_hat = work[(a_hat[0], 0)]
    Uc = work[(U, 2)]
    Uc_hat = CDD.matvec(a_hat[0], Uc_hat)
    Uc_hat = TDMASolverD(Uc_hat)
    dudx = Uc[0] = FST.backward(Uc_hat, Uc[0], ST)
    dvdy_h = 1j*K[1]*a_hat[1]
    dvdy = Uc[1] = FST.backward(dvdy_h, Uc[1], ST)
    dwdz_h = 1j*K[2]*a_hat[2]
    dwdz = Uc[2] = FST.backward(dwdz_h, Uc[2], ST)
    return dudx+dvdy+dwdz

def Divu(U_hat, c):
    c[:] = 0
    LUsolve.Mult_Div_3D(params.N[0], K[1, 0], K[2, 0], U_hat[0, u_slice],
                        U_hat[1, u_slice], U_hat[2, u_slice], c[p_slice])
    c = TDMASolverN(c)
    return c

#@profile
def standardConvection(c, U, U_hat, K, FST, ST, work, mat, la):
    c[:] = 0
    Uc = work[(U, 1)]
    Uc2 = work[(U, 2)]
    F_tmp = work[(U_hat, 0)]

    # dudx = 0 from continuity equation. Use Shen Dirichlet basis
    # Use regular Chebyshev basis for dvdx and dwdx
    F_tmp[0] = mat.CDD.matvec(U_hat[0], F_tmp[0])
    F_tmp[0] = la.TDMASolverD(F_tmp[0])
    dudx = Uc[0] = FST.backward(F_tmp[0], Uc[0], ST, dealias=params.dealias)

    #F_tmp[0] = mat.CND.matvec(U_hat[0])
    #F_tmp[0] = la.TDMASolverN(F_tmp[0])
    #quad = SN.quad
    #SN.quad = ST.quad
    #dudx = Uc[0] = FST.backward(F_tmp[0], Uc[0], SN, dealias=params.dealias)

    LUsolve.Mult_CTD_3D_n(params.N[0], U_hat[1], U_hat[2], F_tmp[1], F_tmp[2])
    dvdx = Uc[1] = FST.backward(F_tmp[1], Uc[1], ST.CT, dealias=params.dealias)
    dwdx = Uc[2] = FST.backward(F_tmp[2], Uc[2], ST.CT, dealias=params.dealias)

    #dudx = U_tmp[0] = chebDerivative_3D0(U[0], U_tmp[0])
    #dvdx = U_tmp[1] = chebDerivative_3D0(U[1], U_tmp[1])
    #dwdx = U_tmp[2] = chebDerivative_3D0(U[2], U_tmp[2])

    dudy = Uc2[0] = FST.backward(1j*K[1]*U_hat[0], Uc2[0], ST, dealias=params.dealias)
    dudz = Uc2[1] = FST.backward(1j*K[2]*U_hat[0], Uc2[1], ST, dealias=params.dealias)
    c[0] = FST.scalar_product(U[0]*dudx + U[1]*dudy + U[2]*dudz, c[0], ST, dealias=params.dealias)

    Uc2[:] = 0
    dvdy = Uc2[0] = FST.backward(1j*K[1]*U_hat[1], Uc2[0], ST, dealias=params.dealias)
    #F_tmp[0] = FST.forward(U[1], F_tmp[0], SN)
    #dvdy_h = 1j*K[1]*F_tmp[0]
    #dvdy = U_tmp2[0] = FST.backward(dvdy_h, U_tmp2[0], SN)
    ##########

    dvdz = Uc2[1] = FST.backward(1j*K[2]*U_hat[1], Uc2[1], ST, dealias=params.dealias)
    c[1] = FST.scalar_product(U[0]*dvdx + U[1]*dvdy + U[2]*dvdz, c[1], ST, dealias=params.dealias)

    Uc2[:] = 0
    dwdy = Uc2[0] = FST.backward(1j*K[1]*U_hat[2], Uc2[0], ST, dealias=params.dealias)

    dwdz = Uc2[1] = FST.backward(1j*K[2]*U_hat[2], Uc2[1], ST, dealias=params.dealias)

    #F_tmp[0] = FST.forward(U[2], F_tmp[0], SN)
    #dwdz_h = 1j*K[2]*F_tmp[0]
    #dwdz = U_tmp2[1] = FST.backward(dwdz_h, U_tmp2[1], SN)
    #########
    c[2] = FST.scalar_product(U[0]*dwdx + U[1]*dwdy + U[2]*dwdz, c[2], ST, dealias=params.dealias)

    # Reset
    #SN.quad = quad

    return c

def divergenceConvection(c, U, U_hat, K, FST, ST, work, mat, la, add=False):
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

    F_tmp[0] = FST.forward(U[0]*U[0], F_tmp[0], ST, dealias=params.dealias)
    F_tmp[1] = FST.forward(U[0]*U[1], F_tmp[1], ST, dealias=params.dealias)
    F_tmp[2] = FST.forward(U[0]*U[2], F_tmp[2], ST, dealias=params.dealias)

    c[0] += mat.CDD.matvec(F_tmp[0], F_tmp2[0])
    c[1] += mat.CDD.matvec(F_tmp[1], F_tmp2[1])
    c[2] += mat.CDD.matvec(F_tmp[2], F_tmp2[2])

    F_tmp2[0] = FST.scalar_product(U[0]*U[1], F_tmp2[0], ST, dealias=params.dealias)
    F_tmp2[1] = FST.scalar_product(U[0]*U[2], F_tmp2[1], ST, dealias=params.dealias)
    c[0] += 1j*K[1]*F_tmp2[0] # duvdy
    c[0] += 1j*K[2]*F_tmp2[1] # duwdz

    F_tmp[0] = FST.scalar_product(U[1]*U[1], F_tmp[0], ST, dealias=params.dealias)
    F_tmp[1] = FST.scalar_product(U[1]*U[2], F_tmp[1], ST, dealias=params.dealias)
    F_tmp[2] = FST.scalar_product(U[2]*U[2], F_tmp[2], ST, dealias=params.dealias)
    c[1] += 1j*K[1]*F_tmp[0]  # dvvdy
    c[1] += 1j*K[2]*F_tmp[1]  # dvwdz
    c[2] += 1j*K[1]*F_tmp[1]  # dvwdy
    c[2] += 1j*K[2]*F_tmp[2]  # dwwdz
    return c

def getConvection(convection):
    if convection == "Standard":

        def Conv(rhs, u_hat, K, FST, ST, work, mat, la):

            u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            for i in range(3):
                u_dealias[i] = FST.backward(u_hat[i], u_dealias[i], ST, params.dealias)

            rhs = standardConvection(rhs, u_dealias, u_hat, K, FST, ST, work, mat, la)
            rhs *= -1
            return rhs

    elif convection == "Divergence":

        def Conv(rhs, u_hat, K, FST, ST, work, mat, la):

            u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            for i in range(3):
                u_dealias[i] = FST.backward(u_hat[i], u_dealias[i], ST, params.dealias)

            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, FST, ST, work, mat, la, False)
            rhs *= -1
            return rhs

    elif convection == "Skew":

        def Conv(rhs, u_hat, K, FST, ST, work, mat, la):

            u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            for i in range(3):
                u_dealias[i] = FST.backward(u_hat[i], u_dealias[i], ST, params.dealias)

            rhs = standardConvection(rhs, u_dealias, u_hat, K, FST, ST, work, mat, la)
            rhs = divergenceConvection(rhs, u_dealias, u_hat, K, FST, ST, work, mat, la, True)
            rhs *= -0.5
            return rhs

    elif convection == "Vortex":

        def Conv(rhs, u_hat, K, FST, ST, work, mat, la):

            u_dealias = work[((3,)+FST.work_shape(params.dealias), float, 0)]
            curl_dealias = work[((3,)+FST.work_shape(params.dealias), float, 1)]
            for i in range(3):
                u_dealias[i] = FST.backward(u_hat[i], u_dealias[i], ST, params.dealias)

            curl_dealias[:] = compute_curl(curl_dealias, u_hat,K, FST, ST, work)
            rhs = Cross(rhs, u_dealias, curl_dealias, FST, ST, work)
            return rhs

    Conv.convection = convection
    return Conv

def ComputeRHS(rhs, u_hat, p_hat, jj, solver, H_hat, H_hat0, H_hat1, diff0, la, mat, K,
               work, FST, ST, Nu, Sk, **context):

    # Add convection to rhs
    if jj == 0:
        H_hat = solver.conv(H_hat, u_hat, K, FST, ST, work, mat, la)

        # Compute diffusion
        diff0[0] = mat.AB.matvec(u_hat[0], diff0[0])
        diff0[1] = mat.AB.matvec(u_hat[1], diff0[1])
        diff0[2] = mat.AB.matvec(u_hat[2], diff0[2])

        H_hat0[:] = 1.5*H_hat - 0.5*H_hat1

    rhs[:] = H_hat0

    # Add pressure gradient and body force
    rhs = solver.pressuregrad(rhs, p_hat, mat, work, K, Nu)
    rhs = solver.body_force(rhs, Sk, Nu)

    # Scale by 2/nu factor
    rhs *= 2./params.nu

    # Add diffusion
    rhs += diff0

    return rhs

def solve_tentative(u_hat, p_hat, rhs, jj, solver, context):
    rhs[:] = 0
    rhs[:] = solver.ComputeRHS(rhs, u_hat, p_hat, jj, solver, **context)
    u_hat[0] = context.la.HelmholtzSolverU(u_hat[0], rhs[0])
    u_hat[1] = context.la.HelmholtzSolverU(u_hat[1], rhs[1])
    u_hat[2] = context.la.HelmholtzSolverU(u_hat[2], rhs[2])
    return u_hat

def solve_pressure_correction(p_hat, u_hat, solver,
                              Pcorr, K, la, work, u_slice, p_slice, **context):
    dP = work[(p_hat, 0)]
    dP = solver.pressurerhs(dP, u_hat, K, u_slice, p_slice)
    Pcorr[:] = la.HelmholtzSolverP(Pcorr, dP)
    # Update pressure
    p_hat[p_slice] += Pcorr[p_slice]
    return p_hat, Pcorr

def update_velocity(u_hat, p_corr, rhs, solver, mat, work, K, Nu, u_slice, la, **context):
    rhs[:] = 0
    rhs = solver.pressuregrad(rhs, p_corr, mat, work, K, Nu)
    rhs[0] = la.TDMASolverD(rhs[0])
    rhs[1] = la.TDMASolverD(rhs[1])
    rhs[2] = la.TDMASolverD(rhs[2])
    u_hat[:, u_slice] += params.dt*rhs[:, u_slice]  # + since pressuregrad computes negative pressure gradient
    return u_hat

#@profile
def integrate(u_hat, p_hat, rhs, dt, solver, context):

    pressure_error = zeros(1)
    # Tentative momentum solve
    for jj in range(params.velocity_pressure_iters):

        u_hat = solver.solve_tentative(u_hat, p_hat, rhs, jj, solver, context)

        # Pressure correction
        p_hat, p_corr = solver.solve_pressure_correction(p_hat, u_hat, solver, **context)

        comm.Allreduce([np.linalg.norm(p_corr), MPI.DOUBLE], pressure_error)
        if jj == 0 and params.print_divergence_progress and rank == 0:
            print("   Divergence error")
        if params.print_divergence_progress:
            if rank == 0:
                print("         Pressure correction norm %6d  %2.6e" %(jj, pressure_error[0]))
        if pressure_error[0] < params.divergence_tol:
            break

    # Update velocity
    u_hat = solver.update_velocity(u_hat, p_corr, rhs, solver, **context)

    # Rotate velocities and convection
    context.U_hat1[:] = context.U_hat0
    context.U_hat0[:] = u_hat
    context.H_hat1[:] = context.H_hat

    return (u_hat, p_hat), dt, dt

def getintegrator(rhs, u0, solver, context):
    u_hat, p_hat = u0
    @wraps(solver.integrate)
    def func():
        return solver.integrate(u_hat, p_hat, rhs, params.dt, solver, context)
    return func
