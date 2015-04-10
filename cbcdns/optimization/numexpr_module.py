import numexpr

__all__ = ['add_pressure_diffusion', 'cross1', 'cross2', 'dealias_rhs']

def cross1(c, a, b):
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    c[0] = numexpr.evaluate("a1*b2-a2*b1")
    c[1] = numexpr.evaluate("a2*b0-a0*b2")
    c[2] = numexpr.evaluate("a0*b1-a1*b0")
    return c
    
def cross2(c, a, b):
    cross1(c, a, b)
    c *= 1j
    return c

def dealias_rhs(dU, dealias):
    dU = numexpr.evaluate("dU*dealias")
    return dU

def add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu):
    du0, du1, du2 = dU[0], dU[1], dU[2]
    k_0, k_1, k_2 = K_over_K2[0], K_over_K2[1], K_over_K2[2]
    P_hat[:] = numexpr.evaluate("du0*k_0+du1*k_1+du2*k_2")
    dU[:] = numexpr.evaluate("dU - P_hat*K - nu*K2*U_hat")    
    return dU
