__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-07"
__copyright__ = "Copyright (C) 2014-2018 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

#pylint: disable=unused-variable,unused-argument,function-redefined

# Reuses most of NS.py module, but curl in 2D is a scalar
from .NS import *

NS_context = get_context

def get_context():
    """Set up context for NS2D solver"""
    c = NS_context()
    c.curl = Array(c.T)
    c.W_hat = Function(c.T)
    return c

def get_curl(curl, W_hat, U_hat, work, T, K, **context):
    W_hat[:] = 0
    W_hat = cross2(W_hat, K, U_hat)
    curl = W_hat.backward(curl)
    return curl

def getConvection(convection):
    """Return function used to compute nonlinear term"""
    if convection in ("Standard", "Divergence", "Skewed"):
        raise NotImplementedError

    elif convection == "Vortex":

        def Conv(rhs, u_hat, work, T, Tp, VT, VTp, K):
            u_dealias = work[(VTp.local_shape(False), float, 0)]
            curl_dealias = work[(Tp.local_shape(False), float, 0)]
            curl_hat = work[(Tp.local_shape(True), complex, 0)]

            curl_hat = cross2(curl_hat, K, u_hat)
            curl_dealias = Tp.backward(curl_hat, curl_dealias)
            u_dealias = VTp.backward(u_hat, u_dealias)
            rhs[0] = Tp.forward(u_dealias[1]*curl_dealias, rhs[0])
            rhs[1] = Tp.forward(-u_dealias[0]*curl_dealias, rhs[1])
            return rhs

    Conv.convection = convection
    return Conv
