#pythran export loop1(complex64[:, :, :, :], complex64[:, :, :, :], complex64[:, :, :, :])
#pythran export loop1(complex128[:, :, :, :], complex128[:, :, :, :], complex128[:, :, :, :])
def loop1(U_hat, U_hat0, U_hat1):
    for i in range(U_hat.shape[0]):
        for j in range(U_hat.shape[1]):
            for k in range(U_hat.shape[2]):
                for l in range(U_hat.shape[3]):
                    z = U_hat[i, j, k, l]
                    U_hat1[i, j, k, l] = z
                    U_hat0[i, j, k, l] = z

#pythran export loop2(complex64[:, :, :, :], complex64[:, :, :, :], complex64[:, :, :, :], float32, float32)
#pythran export loop2(complex128[:, :, :, :], complex128[:, :, :, :], complex128[:, :, :, :], float64, float64)
def loop2(dU, U_hat, U_hat0, b, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat[i, j, k, l] = U_hat0[i, j, k, l] + b*dt*dU[i, j, k, l]

#pythran export loop3(complex64[:, :, :, :], complex64[:, :, :, :], float32, float32)
#pythran export loop3(complex128[:, :, :, :], complex128[:, :, :, :], float64, float64)
def loop3(dU, U_hat1, a, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat1[i, j, k, l] = U_hat1[i, j, k, l] + a*dt*dU[i, j, k, l]

#pythran export loop4(complex64[:, :, :, :], complex64[:, :, :, :])
#pythran export loop4(complex128[:, :, :, :], complex128[:, :, :, :])
def loop4(U_hat, U_hat1):
    for i in range(U_hat.shape[0]):
        for j in range(U_hat.shape[1]):
            for k in range(U_hat.shape[2]):
                for l in range(U_hat.shape[3]):
                    U_hat[i, j, k, l] = U_hat1[i, j, k, l]

#pythran export loop5(complex64[:, :, :, :], complex64[:, :, :, :], float32)
#pythran export loop5(complex128[:, :, :, :], complex128[:, :, :, :], float64)
def loop5(dU, U_hat, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat[i, j, k, l] += dU[i, j, k, l]*dt

#pythran export loop6(complex64[:, :, :, :], complex64[:, :, :, :], complex64[:, :, :, :], float32)
#pythran export loop6(complex128[:, :, :, :], complex128[:, :, :, :], complex128[:, :, :, :], float64)
def loop6(dU, U_hat, U_hat0, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat[i, j, k, l] = U_hat[i, j, k, l] + 1.5*dU[i, j, k, l]*dt - 0.5*U_hat0[i, j, k, l]

#pythran export loop7(complex64[:, :, :, :], complex64[:, :, :, :], float32)
#pythran export loop7(complex128[:, :, :, :], complex128[:, :, :, :], float64)
def loop7(dU, U_hat0, dt):
    for i in range(dU.shape[0]):
        for j in range(dU.shape[1]):
            for k in range(dU.shape[2]):
                for l in range(dU.shape[3]):
                    U_hat0[i, j, k, l] = dU[i, j, k, l]*dt

#pythran export cross1(float32[:, :, :, :], float32[:, :, :, :], float32[:, :, :, :])
#pythran export cross1(float64[:, :, :, :], float64[:, :, :, :], float64[:, :, :, :])
def cross1(c, a, b):
    """Regular c = a x b"""
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            for k in range(a.shape[3]):
                a0 = a[0, i, j, k]
                a1 = a[1, i, j, k]
                a2 = a[2, i, j, k]
                b0 = b[0, i, j, k]
                b1 = b[1, i, j, k]
                b2 = b[2, i, j, k]
                c[0, i, j, k] = a1*b2 - a2*b1
                c[1, i, j, k] = a2*b0 - a0*b2
                c[2, i, j, k] = a0*b1 - a1*b0
    return c

#pythran export cross2a(complex64[:, :, :, :], float32[:, :, :, :], complex64[:, :, :, :])
#pythran export cross2a(complex128[:, :, :, :], float64[:, :, :, :], complex128[:, :, :, :])
def cross2a(c, a, b):
    """ c = 1j*(a x b)"""
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            for k in range(a.shape[3]):
                a0 = a[0, i, j, k]
                a1 = a[1, i, j, k]
                a2 = a[2, i, j, k]
                b0 = b[0, i, j, k]
                b1 = b[1, i, j, k]
                b2 = b[2, i, j, k]
                c[0, i, j, k] = -(a1*b2.imag - a2*b1.imag) + 1j*(a1*b2.real - a2*b1.real)
                c[1, i, j, k] = -(a2*b0.imag - a0*b2.imag) + 1j*(a2*b0.real - a0*b2.real)
                c[2, i, j, k] = -(a0*b1.imag - a1*b0.imag) + 1j*(a0*b1.real - a1*b0.real)
    return c

#pythran export cross2c(complex64[:, :, :, :], float32[:], float32[:], float32[:], complex64[:, :, :, :])
#pythran export cross2c(complex128[:, :, :, :], float64[:], float64[:], float64[:], complex128[:, :, :, :])
def cross2c(c, a0, a1, a2, b):
    """ c = 1j*(a x b)"""
    for i in range(b.shape[1]):
        for j in range(b.shape[2]):
            for k in range(b.shape[3]):
                a00 = a0[i]
                a11 = a1[j]
                a22 = a2[k]
                b0 = b[0, i, j, k]
                b1 = b[1, i, j, k]
                b2 = b[2, i, j, k]
                c[0, i, j, k] = -(a11*b2.imag - a22*b1.imag) + 1j*(a11*b2.real - a22*b1.real)
                c[1, i, j, k] = -(a22*b0.imag - a00*b2.imag) + 1j*(a22*b0.real - a00*b2.real)
                c[2, i, j, k] = -(a00*b1.imag - a11*b0.imag) + 1j*(a00*b1.real - a11*b0.real)
    return c

#pythran export add_pressure_diffusion_NS_(complex128[:, :, :, :], complex128[:, :, :, :], float64, float64[:, :, :], float64[:], float64[:], float64[:], complex128[:, :, :], float64[:, :, :, :])
#pythran export add_pressure_diffusion_NS_(complex64[:, :, :, :], complex64[:, :, :, :], float32, float32[:, :, :], float32[:], float32[:], float32[:], complex64[:, :, :], float32[:, :, :, :])
def add_pressure_diffusion_NS_(du, u_hat, nu, ksq, kx, ky, kz, p_hat, k_over_k2):
    for i in range(ksq.shape[0]):
        k0 = kx[i]
        for j in range(ksq.shape[1]):
            k1 = ky[j]
            for k in range(ksq.shape[2]):
                z = nu*ksq[i, j, k]
                k2 = kz[k]
                p_hat[i, j, k] = du[0, i, j, k]*k_over_k2[0, i, j, k]+du[1, i, j, k]*k_over_k2[1, i, j, k]+du[2, i, j, k]*k_over_k2[2, i, j, k]
                du[0, i, j, k] = du[0, i, j, k] - (p_hat[i, j, k]*k0+u_hat[0, i, j, k]*z)
                du[1, i, j, k] = du[1, i, j, k] - (p_hat[i, j, k]*k1+u_hat[1, i, j, k]*z)
                du[2, i, j, k] = du[2, i, j, k] - (p_hat[i, j, k]*k2+u_hat[2, i, j, k]*z)
    return du

#pythran export compute_vw(complex128[:, :, :, :], complex128[:, :, :], complex128[:, :, :], float64[:, :, :, :])
#pythran export compute_vw(complex64[:, :, :, :], complex64[:, :, :], complex64[:, :, :], float32[:, :, :, :])
def compute_vw(u_hat, f_hat, g_hat, k_over_k2):
    for i in range(u_hat.shape[1]):
        for j in range(u_hat.shape[2]):
            for k in range(u_hat.shape[3]):
                u_hat[1, i, j, k] = -1j*(k_over_k2[0, i, j, k]*f_hat[i, j, k] - k_over_k2[1, i, j, k]*g_hat[i, j, k])
                u_hat[2, i, j, k] = -1j*(k_over_k2[1, i, j, k]*f_hat[i, j, k] + k_over_k2[0, i, j, k]*g_hat[i, j, k])
    return u_hat

#pythran export _mult_K1j(float64[:], float64[:], complex128[:, :, :], complex128[:, :, :, :])
def _mult_K1j(Ky, Kz, a, f):
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                f[0, i, j, k] = 1j*Kz[k]*a[i, j, k]
                f[1, i, j, k] = -1j*Ky[j]*a[i, j, k]
    return f
