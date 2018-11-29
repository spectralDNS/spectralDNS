from .pythran_maths import loop1, loop2, loop3, loop4, loop5, loop6, loop7, \
    cross1, cross2a, cross2c, add_pressure_diffusion_NS_, _mult_K1j, compute_vw

def RK4(U_hat, U_hat0, U_hat1, dU, a, b, dt, solver, context):
    loop1(U_hat, U_hat0, U_hat1)
    c = context
    for rk in range(4):
        dU = solver.ComputeRHS(dU, U_hat, solver, **c)
        if rk < 3:
            loop2(dU, U_hat, U_hat0, b[rk], dt)
        loop3(dU, U_hat1, a[rk], dt)
    loop4(U_hat, U_hat1)
    return U_hat, dt, dt

def ForwardEuler(U_hat, dU, dt, solver, context):
    dU = solver.ComputeRHS(dU, U_hat, solver, **context)
    loop5(dU, U_hat, dt)
    return U_hat, dt, dt

def AB2(U_hat, U_hat0, dU, dt, tstep, solver, context):
    dU = solver.ComputeRHS(dU, U_hat, solver, **context)
    if tstep == 0:
        loop5(dU, U_hat, dt)
    else:
        loop6(dU, U_hat, U_hat0, dt)
    loop7(dU, U_hat0, dt)
    return U_hat, dt, dt

def cross2(c, a, b):
    if isinstance(a, list):
        c = cross2c(c, a[0][:, 0, 0], a[1][0, :, 0], a[2][0, 0, :], b)
    else:
        c = cross2a(c, a, b)
    return c

def add_pressure_diffusion_NS(du, u_hat, nu, ksq, kk, p_hat, k_over_k2):
    du = add_pressure_diffusion_NS_(du, u_hat, nu, ksq, kk[0][:, 0, 0],
                                    kk[1][0, :, 0], kk[2][0, 0, :], p_hat, k_over_k2)
    return du

def mult_K1j(K, a, f):
    f = _mult_K1j(K[1][0, :, 0], K[2][0, 0], a, f)
    return f
