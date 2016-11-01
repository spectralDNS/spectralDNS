from numpy import *
from spectralDNS.shen.shentransform import ShenBiharmonicBasis, ShenDirichletBasis
from spectralDNS.shen.la import Helmholtz, TDMA, Biharmonic

nu = 1./5200.
dt = 0.00001

#HelmholtzSolverG = Helmholtz(N[0], np.sqrt(K2[0]+2.0/nu/dt), ST.quad, False),
#BiharmonicSolverU = Biharmonic(N[0], -nu*dt/2., 1.+nu*dt*K2[0],
                                    #-(K2[0] + nu*dt/2.*K4[0]), quad=SB.quad,
                                    #solver="cython")

N = array([64, 128, 256, 512, 1024, 2048, 4096])
Z = array([0, 200, 1800, 5400])
M = 100

print "\hline"
print "z & " + " & ".join([str(n) for n in N]) + " \\\ "
print "\hline"
for z in Z:
    err = str(z) 
    for n in N:
        errb = 0
        vb = zeros(n)
        sb = zeros(n)
        BH = Biharmonic(n, -nu*dt/2., 1.+nu*dt*z**2, -(z**2+nu*dt/2.*z**4), "GC", "cython")
        for m in range(M):
            u = random.random(n)
            u[-4:] = 0
            vb = BH.matvec(u, vb)
            sb = BH(sb, vb)
            errb += max(abs(sb-u)) / max(abs(u))
        err += " & {:2.2e} ".format(errb/M)
    err += " \\\ "
    print err

print "\hline"        
for z in Z:
    err = str(z)
    for n in N:
        errh = 0
        vh = zeros(n)
        sh = zeros(n)
        alfa = sqrt(z**2+2.0/nu/dt)
        HS = Helmholtz(n, alfa, "GC")
        for m in range(M):
            u = random.randn(n)
            u[-2:] = 0
            vh = HS.matvec(u, vh)
            sh = HS(sh, vh)
            errh += max(abs(sh-u)) / max(abs(u))
        err += " & {:2.2e} ".format(errh/M)
    err += " \\\ "
    print err


        