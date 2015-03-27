import numexpr

__all__ = ['numexpr_add_pressure_diffusion', 'numexpr_cross1', 'numexpr_cross2', 'numexpr_dealias_rhs']

def numexpr_cross1(a, b, c):
    """Regular c = a x b"""
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    c[0] = numexpr.evaluate("a1*b2-a2*b1")
    c[1] = numexpr.evaluate("a2*b0-a0*b2")
    c[2] = numexpr.evaluate("a0*b1-a1*b0")
    return c
    
def numexpr_cross2(a, b, c):
    """ c = 1j*(a x b)"""
    numexpr_cross1(a, b, c)
    c *= 1j
    return c

def numexpr_dealias_rhs(dU, dealias):
    """Dealias the nonlinear convection"""
    dU = numexpr.evaluate("dU*dealias")
    return dU

def numexpr_add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu):
    """Add contributions from pressure and diffusion to the rhs"""
    
    # Compute pressure (To get actual pressure multiply by 1j)
    du0, du1, du2 = dU[0], dU[1], dU[2]
    k0, k1, k2 = K[0], K[1], K[2]
    u0, u1, u2 = U_hat[0], U_hat[1], U_hat[2]
    dU = numexpr.evaluate("(du0+du1+du2)*K_over_K2 - P_hat*(k0+k1+k2) - nu*K2*(u0+u1+u2)")            
    return dU
