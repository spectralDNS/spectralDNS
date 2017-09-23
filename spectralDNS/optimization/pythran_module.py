from pythran_maths import cross1, cross2 as cross2_, add_pressure_diffusion_NS as add_pressure_diffusion_NS_
from numpy import asarray

def cross2(c, a, b):
    return cross2_(c, asarray(a), b)

def add_pressure_diffusion_NS(du, u_hat, nu, ksq, kk, p_hat, k_over_k2):
    return add_pressure_diffusion_NS_(du, u_hat, nu, ksq, asarray(kk), p_hat, k_over_k2)
