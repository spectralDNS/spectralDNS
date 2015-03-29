__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-03-24"
__copyright__ = "Copyright (C) 2015 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

transpose_Uc, transpose_Umpi = None, None
transform_Uc_xz, transform_Uc_yx, transform_Uc_xy, transform_Uc_zx = None, None, None, None

try:
    import cython_single, cython_double
    def cython_add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu):
        if dU.dtype == "complex64":
            cython_single.cython_add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
        else:
            cython_double.cython_add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
        return dU
        
    def cython_cross1(a, b, c):
        if c.dtype == "float32":
            cython_single.cython_cross1(a, b, c)
        else:
            cython_double.cython_cross1(a, b, c)
        return c
            
    def cython_cross2(a, b, c):
        if c.dtype == "complex64":
            cython_single.cython_cross2(a, b, c)
        else:
            cython_double.cython_cross2(a, b, c)
        return c

    def cython_dealias_rhs(a, b):
        if a.dtype == "complex64":
            cython_single.cython_dealias(a, b)
        else:
            cython_double.cython_dealias(a, b)
        return a
    
    def transpose_Uc(a, b, c, d):
        if a.dtype == "complex64":
            cython_single.transpose_Uc(a, b, c, d)
        else:
            cython_double.transpose_Uc(a, b, c, d)
        return a

    def transpose_Umpi(a, b, c, d):
        if a.dtype == "complex64":
            cython_single.transpose_Umpi(a, b, c, d)
        else:
            cython_double.transpose_Umpi(a, b, c, d)
        return a

    def transform_Uc_xz(a, b, c, d):
        if a.dtype == "complex64":
            cython_single.transform_Uc_xz(a, b, c, d)
        else:
            cython_double.transform_Uc_xz(a, b, c, d)
        return a

    def transform_Uc_yx(a, b, c, d):
        if a.dtype == "complex64":
            cython_single.transform_Uc_yx(a, b, c, d)
        else:
            cython_double.transform_Uc_yx(a, b, c, d)
        return a

    def transform_Uc_xy(a, b, c, d):
        if a.dtype == "complex64":
            cython_single.transform_Uc_xy(a, b, c, d)
        else:
            cython_double.transform_Uc_xy(a, b, c, d)
        return a

    def transform_Uc_zx(a, b, c, d):
        if a.dtype == "complex64":
            cython_single.transform_Uc_zx(a, b, c, d)
        else:
            cython_double.transform_Uc_zx(a, b, c, d)
        return a
            
except:
    pass

try:
    import weave_single, weave_double
    def weave_cross1(a, b, c):
        if c.dtype == "float32":
            weave_single.weave_cross1(a, b, c)
        else:
            weave_double.weave_cross1(a, b, c)
        return c
            
    def weave_cross2(a, b, c):
        if c.dtype == "complex64":
            weave_single.weave_cross2(a, b, c)
        else:
            weave_double.weave_cross2(a, b, c)
        return c

    def weave_add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu):
        if dU.dtype == "complex64":
            weave_single.weave_add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
        else:
            weave_double.weave_add_pressure_diffusion(dU, U_hat, K2, K, P_hat, K_over_K2, nu)
        return dU
            
    def weave_dealias_rhs(a, b):
        if a.dtype == "complex64":
            weave_single.weave_dealias(a, b)
        else:
            weave_double.weave_dealias(a, b)
        return a
    
except:
    pass

try:   
    import numba_single, numba_double
    def numba_dealias_rhs(du, dealias):
        if du.dtype == 'complex64':
            du = numba_single.dealias(du, dealias)
        else:
            du = numba_double.dealias(du, dealias)
        return du

    def numba_cross1(a, b, c):
        if c.dtype == "float32":
            c = numba_single.cross1(a, b, c)
        else:
            c = numba_double.cross1(a, b, c)
        return c

    def numba_cross2(a, b, c):
        if c.dtype == "complex64":
            c = numba_single.cross2(a, b, c)
        else:
            c = numba_double.cross2(a, b, c)
        return c

    def numba_add_pressure_diffusion(du, u_hat, ksq, kk, p_hat, k_over_k2, nu):
        if du.dtype == "complex64":
            du = numba_single.add_pressure_diffusion(du, u_hat, ksq, kk, p_hat, k_over_k2, nu)
        else:
            du = numba_double.add_pressure_diffusion(du, u_hat, ksq, kk, p_hat, k_over_k2, nu)
        return du

    def transpose_Uc(a, b, c, d):
        if a.dtype == "complex64":
            a = numba_single.transpose_Uc(a, b, c, d)
        else:
            a = numba_double.transpose_Uc(a, b, c, d)
        return a

    def transpose_Umpi(a, b, c, d):
        if a.dtype == "complex64":
            a = numba_single.transpose_Umpi(a, b, c, d)
        else:
            a = numba_double.transpose_Umpi(a, b, c, d)
        return a

    def transform_Uc_xz(a, b, c, d):
        if a.dtype == "complex64":
            a = numba_single.transform_Uc_xz(a, b, c, d)
        else:
            a = numba_double.transform_Uc_xz(a, b, c, d)
        return a

    def transform_Uc_yx(a, b, c, d):
        if a.dtype == "complex64":
            a = numba_single.transform_Uc_yx(a, b, c, d)
        else:
            a = numba_double.transform_Uc_yx(a, b, c, d)
        return a

    def transform_Uc_xy(a, b, c, d):
        if a.dtype == "complex64":
            a = numba_single.transform_Uc_xy(a, b, c, d)
        else:
            a = numba_double.transform_Uc_xy(a, b, c, d)
        return a

    def transform_Uc_zx(a, b, c, d):
        if a.dtype == "complex64":
            a = numba_single.transform_Uc_zx(a, b, c, d)
        else:
            a = numba_double.transform_Uc_zx(a, b, c, d)
        return a
    
except:
    pass

try:
    from numexpr_module import *

except:
    pass

    