"""
Velocity-vorticity formulation

Derived by taking the curl of the momentum equation and solving for the curl.
The velocity that is required in the convective term is computed from the curl
in the compute_velocity method.

This solver inherits most features from the NS solver.
Overloading just a few routines.

"""
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-01-02"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-variable,unused-argument,function-redefined

from .NS import *

NS_context = get_context

def get_context():
    """Set up context for Velocity-Vorticity (VV) solver"""
    c = NS_context()
    del c.P, c.P_hat
    c.W_hat = empty((3,) + c.FFT.complex_shape(), dtype=complex) # curl transformed
    c.u = c.W_hat
    c.hdf5file = VVWriter({'U':c.U[0], 'V':c.U[1], 'W':c.U[2],
                           'curlx':c.curl[0], 'curly':c.curl[1], 'curlz':c.curl[2]},
                          chkpoint={'current':{'U':c.U, 'curl':c.curl}, 'previous':{}},
                          filename=params.h5filename+'.h5')
    return c

class VVWriter(HDF5Writer):
    """Subclass HDF5Writer for appropriate updating of real components"""
    def update_components(self, **context):
        """Transform to real data when storing the solution"""
        get_velocity(**context)
        get_curl(**context)

def compute_velocity(c, w_hat, work, FFT, K_over_K2, dealias=None):
    """Compute u from curl(u)

    Follows from
      w = [curl(u)=] \nabla \times u
      curl(w) = \nabla^2(u) (since div(u)=0)
      FFT(curl(w)) = FFT(\nabla^2(u))
      ik \times w_hat = k^2 u_hat
      u_hat = (ik \times w_hat) / k^2
      u = iFFT(u_hat)

    """
    v_hat = work[(w_hat, 0)]
    v_hat = cross2(v_hat, K_over_K2, w_hat)
    c[0] = FFT.ifftn(v_hat[0], c[0], dealias)
    c[1] = FFT.ifftn(v_hat[1], c[1], dealias)
    c[2] = FFT.ifftn(v_hat[2], c[2], dealias)
    return c

def get_velocity(W_hat, U, work, FFT, K_over_K2, **context):
    """Compute velocity from context"""
    U = compute_velocity(U, W_hat, work, FFT, K_over_K2)
    return U

def get_divergence(FFT, K, U_hat, W_hat, **context):
    div_u = zeros(FFT.real_shape())
    U_hat = cross2(U_hat, K, W_hat)
    div_u = FFT.ifftn(1j*(K[0]*U_hat[0]+K[1]*U_hat[1]+K[2]*U_hat[2]), div_u)
    return div_u

def get_curl(curl, W_hat, FFT, **context):
    """Compute curl from context"""
    for i in range(3):
        curl[i] = FFT.ifftn(W_hat[i], curl[i])
    return curl

def getConvection(convection):
    """Return function used to compute nonlinear term"""
    if convection in ("Standard", "Divergence", "Skewed"):
        raise NotImplementedError

    elif convection == "Vortex":

        def Conv(rhs, w_hat, work, FFT, K, K_over_K2):
            u_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 0)]
            w_dealias = work[((3,)+FFT.work_shape(params.dealias), float, 1)]
            v_hat = work[(rhs, 0)]

            u_dealias = compute_velocity(u_dealias, w_hat, work, FFT, K_over_K2, params.dealias)
            for i in range(3):
                w_dealias[i] = FFT.ifftn(w_hat[i], w_dealias[i], params.dealias)
            v_hat = Cross(v_hat, u_dealias, w_dealias, work, FFT, params.dealias) # v_hat = F_k(u_dealias x w_dealias)
            rhs = cross2(rhs, K, v_hat)  # rhs = 1j*(K x v_hat)
            return rhs

    Conv.convection = convection
    return Conv

@optimizer
def add_linear(rhs, w_hat, nu, K2, Source):
    """Add contributions from source and diffusion to the rhs"""
    rhs -= nu*K2*w_hat
    rhs += Source
    return rhs

def ComputeRHS(rhs, w_hat, solver, work, FFT, K, Kx, K2, K_over_K2, Source, **context):
    """Return right hand side of Navier Stokes in velocity-vorticity form

    args:
        rhs         The right hand side to be returned
        w_hat       The FFT of the curl at current time. May differ from
                    context.W_hat since it is set by the integrator
        solver      The solver module. Included for possible inheritance
                    and flexibility of integrators.

    Remaining args may be extracted from context:
        work        Work arrays
        FFT         Transform class from mpiFFT4py
        K           Scaled wavenumber mesh
        Kx          Scaled wavenumber mesh with zero Nyquist frequency
        K2          K[0]*K[0] + K[1]*K[1] + K[2]*K[2]
        K_over_K2   K / K2
        Source      Scalar source term

    """
    rhs = solver.conv(rhs, w_hat, work, FFT, K, K_over_K2)
    rhs = solver.add_linear(rhs, w_hat, params.nu, K2, Source)
    return rhs
