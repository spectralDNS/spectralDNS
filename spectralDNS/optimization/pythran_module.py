from .pythran_maths import cross1_, cross2_, add_pressure_diffusion_NS as add_pressure_diffusion_NS_

def cross2(c, a, b):
    return cross2_(c, a[0], a[1], a[2], b)

def cross1(c, a, b):
    return cross1_(c, a[0], a[1], a[2], b)

def add_pressure_diffusion_NS(du, u_hat, nu, ksq, kk, p_hat, k_over_k2):
    return add_pressure_diffusion_NS_(du, u_hat, nu, ksq, kk[0], kk[1],
                                      kk[2], p_hat, k_over_k2)
