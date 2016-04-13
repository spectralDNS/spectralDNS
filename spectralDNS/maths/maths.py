__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..optimization import optimizer

__all__ = ['dealias_rhs', 'project']

@optimizer
def dealias_rhs(dU, dealias):
    """Dealias the nonlinear convection"""
    dU *= dealias
    return dU

def project(u, K, K_over_K2):
    """Project u onto divergence free space"""
    u -= sum(K_over_K2*u, 0)*K
    return u

