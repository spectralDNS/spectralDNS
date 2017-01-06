from numpy import *
from spectralDNS.shen.shentransform import ShenBiharmonicBasis, ShenDirichletBasis
from spectralDNS.shen.la import Helmholtz, TDMA, Biharmonic
from time import time, sleep
import cProfile, pstats
from six import iteritems

nu = 1./1200.
dt = 0.0001

N = array([64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=int)
#N = array([64, 128, 256, 512], dtype=int)
M = 100

Z = array([0, 32, 32], dtype=int)
ky, kz = (fft.fftfreq(Z[1], 1./Z[1]),
          fft.rfftfreq(Z[2], 1./Z[2]))
K = array(meshgrid(ky, kz, indexing='ij'), dtype=float)
K2 = sum(K*K, 0, dtype=float)
K4 = K2**2

ST = ShenDirichletBasis("GL")
SB = ShenBiharmonicBasis("GL")

t11 = 1
t22 = 1
print("\hline")
print("Nx & Biharmonic & Helmholtz \\\ ")
print("\hline")
#s = StringIO.StringIO()
#sortby = 'cumulative'

#@profile
def main():
    for n in N:
        err = str(n)
        Z[0] = n
        HS = Helmholtz(n, sqrt(K2+2.0/nu/dt), "GC", False)
        BS = Biharmonic(n, -nu*dt/2., 1.+nu*dt*K2,
                        -(K2 + nu*dt/2.*K4), quad="GC",
                        solver="cython")

        fb = random.random((Z[0], Z[1], Z[2]//2+1)) + random.random((Z[0], Z[1], Z[2]//2+1))*1j
        fb[-4:] = 0
        ub = zeros((Z[0], Z[1], Z[2]//2+1), dtype=complex)
        #sleep(0.5)
        t0 = time()
        for m in range(M):
            ub = BS(ub, fb)
        t1 = (time()-t0)/M/Z[1:].prod()

        #cProfile.runctx("for m in range(M): ub = BS(ub, fb)", globals(), locals(), "res1.stats")
        #ps = pstats.Stats("res1.stats")
        #for key, val in iteritems(ps.stats):
            #if "Solve_Biharmonic" in key[2]:
                #results = val[3]/M/Z[1:].prod()
                #break
        #t1 = results

        err += " & {:2.2e} ({:2.2f}) ".format(t1, 0 if n == N[0] else t1/t11/2.)
        t11 = t1

        fh = random.random((Z[0], Z[1], Z[2]//2+1)) + random.random((Z[0], Z[1], Z[2]//2+1))*1j
        fh[-2:] = 0
        uh = zeros((Z[0], Z[1], Z[2]//2+1), dtype=complex)
        #sleep(0.5)
        t0 = time()
        for m in range(M):
            uh = HS(uh, fh)
        t2 = (time()-t0)/M/Z[1:].prod()
        #cProfile.runctx("for m in range(M): uh = HS(uh, fh)", globals(), locals(), "res.stats")
        #ps = pstats.Stats("res.stats")
        #for key, val in iteritems(ps.stats):
            #if "Solve_Helmholtz" in key[2]:
                #results = val[3]/M/Z[1:].prod()
                #break
        #t2 = results

        err += "& {:2.2e} ({:2.2f}) \\\ ".format(t2, 0 if n == N[0] else t2/t22/2.)
        t22 = t2
        print(err)


main()
