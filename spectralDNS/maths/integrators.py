__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2015-04-07"
__copyright__ = "Copyright (C) 2015-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from ..optimization import optimizer, wraps
from numpy import array
import spectralDNS.context
import nodepy
import numpy as np

__all__ = ['getintegrator']


def expEuler(context,dU,f,gexp,hphi,t,tstep,dt,kw):
    U_hat = context.mesh_vars["U_hat"]
    U = context.mesh_vars["U"]
    if "additional_callback" in kw:
        kw["additional_callback"](fU_hat=dU,**kw)

    f(context,U,U_hat,dU,0)
    hphi(context,1,U,dU,0,dt)
    gexp(context,U,U_hat,dU,0,dt)
    U_hat[:] += dU
    return U_hat,dt,dt

def imexEXP(context,imex_offset,f,gexp,A,b,bhat,err_order,fY_hat,U_tmp,U_hat_new,sc,err,fsal,fsal_offset,dU,dt,tstep,kw):
    U_hat = context.mesh_vars["U_hat"]
    U = context.mesh_vars["U"]
    FFT = context.FFT
    #Alternate between doing RK step first and Exponential step
    for i in range(2):
        if (i != imex_offset[0]):
            if i == 1:
                for k in range(U.shape[0]):
                    U[k] = FFT.ifftn(U_hat[k], U[k])
            adaptiveRK(context,A,b,bhat,err_order,fY_hat,U_tmp,U_hat_new,sc,err,fsal,fsal_offset,100,100,False,"2",dU,U_hat,f,dt,tstep,kw)
        else:
            gexp(context,U,U_hat,dU,0,dt)
    imex_offset[0] = (imex_offset[0] + 1) % 2
    return U_hat,dt,dt

def getexpBS5(context,dU,f,gexp):
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]

    A = nodepy.rk.loadRKM("BS5").A.astype(np.float64)
    b = nodepy.rk.loadRKM("BS5").b.astype(np.float64)
    bhat = nodepy.rk.loadRKM("BS5").bhat.astype(np.float64)
    fsal = True

    #Offset for fsal stuff. #TODO: infer this from tstep
    imex_offset = [0]
    fsal_offset = [0]

    s = A.shape[0]
    U_tmp = np.zeros(U.shape, dtype=U.dtype)
    fY_hat = np.zeros((s,) + U_hat.shape, dtype = U_hat.dtype)
    sc = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    err = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    U_hat_new = np.zeros(U_hat.shape,dtype=U_hat.dtype)

    def expBS5(t,tstep,dt,additional_args = {}):
        return imexEXP(context,imex_offset,f,gexp,A,b,bhat,4,fY_hat,U_tmp,U_hat_new,sc,err, fsal,fsal_offset, dU,dt,tstep,additional_args)
    return expBS5


def imexDIRK(context,A,b,A_hat,b_hat,U_tmp,K,K_hat,dU,f,g,ginv,dt,tstep,kw):
    s = A.shape[0] 
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]


    U_tmp[:] = U[:]
    for i in range(s):
        K[i] = U_hat
        for j in range(i):
            K[i] += dt*A[i,j]*K[j][:]
            K[i] += dt*A_hat[i,j]*K_hat[j][:]
        dU = ginv(context,U_tmp,K[i],dU,i,A[i,i]*dt)

        if i == 0 and "additional_callback" in kw:
            kw["additional_callback"](fU_hat=K_hat[i],**kw)

        K_hat[i] = f(context,U_tmp,dU,K_hat[i],i+1)
        K[i] = g(context,U_tmp,dU,K[i],i)
    for i in range(s):
       U_hat[:] += dt*b[i]*K[i]
       U_hat[:] += dt*b_hat[i]*K_hat[i]
    #U_hat[:] += dt*b_hat[s]*K_hat[s]
    return U_hat,dt,dt

def getIMEX1(context,dU,f,g,ginv):
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]

    A = np.array([[1]],dtype=np.float64)
    b = np.array([1],dtype=np.float64)
    A_hat = np.array([[0,0],[1,0]],dtype=np.float64)
    b_hat = np.array([1,0],dtype=np.float64)

    s = A.shape[0] 
    K = np.empty((s,) + U_hat.shape, dtype=U_hat.dtype)
    K_hat = np.empty((s,)+U_hat.shape,dtype=U_hat.dtype)
    U_tmp = np.empty(U.shape,dtype=U.dtype)
#TODO: Do we need to use @wraps here?
    def IMEXOneStep(t,tstep,dt,additional_args = {}):
        return imexDIRK(context,A,b,A_hat,b_hat,U_tmp,K,K_hat,dU,f,g,ginv,dt,tstep,additional_args)
    return IMEXOneStep


def getIMEX4(context,dU,f,g,ginv):
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]

    A = np.array([[0,0,0,0,0],[0,1./2, 0,0,0],[0,1./6,1./2,0,0],[0,-1./2,1./2,1./2,0],[0,3./2,-3./2,1./2,1./2]],dtype=np.float64)
    b= np.array([0,3./2,-3./2,1./2,1./2],dtype=np.float64)

    A_hat = np.array([[0,0,0,0,0],[1./2,0,0,0,0],[11./18,1./18,0,0,0],[5./6,-5./6,1./2,0,0],[1./4,7./4,3./4,-7./4,0]],dtype=np.float64)
    b_hat = np.array([1./4,7./4,3./4,-7./4,0],dtype=np.float64)

    s = A.shape[0] 
    K = np.empty((s,) + U_hat.shape, dtype=U_hat.dtype)
    K_hat = np.empty((s,)+U_hat.shape,dtype=U_hat.dtype)
    U_tmp = np.empty(U.shape,dtype=U.dtype)
#TODO: Do we need to use @wraps here?
    def IMEXOneStep(t,tstep,dt,additional_args = {}):
        return imexDIRK(context,A,b,A_hat,b_hat,U_tmp,K,K_hat,dU,f,g,ginv,dt,tstep,additional_args)
    return IMEXOneStep

def getIMEX3(context,dU,f,g,ginv):
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]

    A_hat = np.array([
        [0,0,0,0],
        [1767732205903./2027836641118,0,0,0],
        [5535828885825./10492691773637,788022342437./10882634858940,0,0],
        [ 6485989280629./16251701735622,-4246266847089./9704473918619,10755448449292./10357097424841,0]
        ],dtype=np.float64)
    b_hat = np.array(
            [1471266399579./7840856788654,-4482444167858./7529755066697,11266239266428./11593286722821,1767732205903./4055673282236,],
            dtype=np.float64)

    A = np.array([
        [0,0,0,0],
        [ 1767732205903./4055673282236,  1767732205903./4055673282236,0,0],
        [2746238789719./10658868560708 , -640167445237./6845629431997,1767732205903./4055673282236,0],
        b_hat
        ],dtype=np.float64)
    b = b_hat


    s = A.shape[0] 
    K = np.empty((s,) + U_hat.shape, dtype=U_hat.dtype)
    K_hat = np.empty((s,)+U_hat.shape,dtype=U_hat.dtype)
    U_tmp = np.empty(U.shape,dtype=U.dtype)
#TODO: Do we need to use @wraps here?
    def IMEXOneStep(t,tstep,dt,additional_args = {}):
        return imexDIRK(context,A,b,A_hat,b_hat,U_tmp,K,K_hat,dU,f,g,ginv,dt,tstep,additional_args)
    return IMEXOneStep

def getIMEX5(context,dU,f,g,ginv):
    U = context.mesh_vars["U"]
    U_hat = context.mesh_vars["U_hat"]

    A_hat = np.array([
        [0,0,0,0,0,0,0,0],
        [41./100,0,0,0,0,0,0,0],
        [367902744464./2072280473677,677623207551./8224143866563,0,0,0,0,0,0],
        [1268023523408./10340822734521,0,1029933939417./13636558850479,0,0,0,0,0],
        [14463281900351./6315353703477,0,66114435211212./5879490589093 , -54053170152839./4284798021562,0,0,0,0],
        [14090043504691./34967701212078,0,15191511035443./11219624916014,-18461159152457./12425892160975,-281667163811./9011619295870,0,0,0],
        [19230459214898./13134317526959,0,21275331358303./2942455364971,-38145345988419./4862620318723,-1./8,-1./8,0,0],
        [-19977161125411./11928030595625,0,-40795976796054./6384907823539 ,177454434618887./12078138498510,782672205425./8267701900261,
            -69563011059811./9646580694205,7356628210526./4942186776405,0]
        ],dtype=np.float64)
    b_hat = np.array(
            [-872700587467./9133579230613,0,0,22348218063261./9555858737531,-1143369518992./8141816002931,-39379526789629./19018526304540,
                32727382324388./42900044865799,41./200], dtype=np.float64)

    A = np.array([
        [0,0,0,0,0,0,0,0],
        [41./200,41./200,0,0,0,0,0,0],
        [41./400,-567603406766./11931857230679,41./200,0,0,0,0,0],
        [683785636431./9252920307686,0,-110385047103./1367015193373,41./200,0,0,0,0],
        [3016520224154./10081342136671,0,30586259806659./12414158314087,-22760509404356./11113319521817,41./200,0,0,0],
        [218866479029./1489978393911,0,638256894668./5436446318841,-1179710474555./5321154724896,-60928119172./8023461067671,41./200,0,0],
        [1020004230633./5715676835656,0,25762820946817./25263940353407,-2161375909145./9755907335909,-211217309593./5846859502534,
            -4269925059573./7827059040749,41./200,0],
        [-872700587467./9133579230613,0,0,22348218063261./9555858737531,-1143369518992./8141816002931,-39379526789629./19018526304540,
            32727382324388./42900044865799,41./200]],dtype=np.float64)
    b = A[-1]


    s = A.shape[0] 
    K = np.empty((s,) + U_hat.shape, dtype=U_hat.dtype)
    K_hat = np.empty((s,)+U_hat.shape,dtype=U_hat.dtype)
    U_tmp = np.empty(U.shape,dtype=U.dtype)
#TODO: Do we need to use @wraps here?
    def IMEXOneStep(t,tstep,dt,additional_args = {}):
        return imexDIRK(context,A,b,A_hat,b_hat,U_tmp,K,K_hat,dU,f,g,ginv,dt,tstep,additional_args)
    return IMEXOneStep



#@optimizer
def adaptiveRK(context,A,b,bhat,err_order, fY_hat,U_tmp,U_hat_new,sc,err, fsal,offset, aTOL,rTOL,adaptive,errnorm,dU,U_hat,ComputeRHS,dt,tstep,kw,predictivecontroller=False):
    """
    Take a step using any Runge-Kutta method.

    Parameters
    ----------
    A, b, bhat : arrays
        Runge-Kutta coefficients
    err_order : int
        Order of embedded method
    fY_hat, U_tmp, U_hat_new, sc, err : work arrays
    fsal : boolean
        Whether method is first-same-as-last
    offset : length-1 array of int
        Where to find the previous RHS evaluation (for FSAL methods).  This can probably be eliminated.
    aTOL, rTOL : float
        Error tolerances
    adaptive : boolean
        If true, adapt the step size
    errnorm : str
        Which norm to use in computing the error estimate.  One of {"2", "inf"}.
    dU : array
        RHS evaluation
    U_hat : array
        solution value (returned)
    ComputeRHS : callable
        RHS of evolution equation
    dt : float
        time step size
    tstep : int
        Number of steps taken so far
    kw : dict
        Information for callback function
    predictivecontroller : boolean
        If True use PI controller
    """
    if not (context.solver_name in ["Bq2D","Bq3D"]):
        U = context.mesh_vars["U"]
    else:
        U = context.mesh_vars["Ur"]
    N = context.model_params["N"]

    FFT = context.FFT
    s = A.shape[0]
    
    #Some parameters for adaptive time-stepping. See p167, Hairer, Norsett and Wanner. "Solving Ordinary Differential Equations 1"
    #for details.
    facmax_default = 2
    facmax = facmax_default
    fac = 0.8
    facmin = 0.01

    #We may need to repeat the time-step until a small enough value is used.
    while True:
        dt_prev = dt
        if fsal:
            offset[0] = (offset[0] - 1) % s
        for i in range(0,s):
            if not fsal or (tstep == 0 or i != 0 ):
                fY_hat[(i + offset[0]) % s] =  U_hat
                for j in range(0,i):
                    fY_hat[(i+offset[0]) % s] += dt*A[i,j]*fY_hat[(j+offset[0]) % s]
                #ComputeRHS does not calculate ifft if i = 0
                if i==0:
                    U_tmp[:] = U 
                #Compute F(Y)
                dU = ComputeRHS(context,U_tmp,fY_hat[(i+offset[0])%s],dU,i)
                fY_hat[(i+offset[0])%s] = dU
            if i == 0 and "additional_callback" in kw:
                kw["additional_callback"](fU_hat=fY_hat[(0+offset[0]) % s],**kw)
 
        #Calculate the new value
        U_hat_new[:] = U_hat
        U_hat_new[:] += dt*b[0]*fY_hat[(0+offset[0])%s]
        err[:] = dt*(b[0] - bhat[0])*fY_hat[(0+offset[0])%s]

        for j in range(1,s):
            U_hat_new[:] += dt*b[j]*fY_hat[(j+offset[0])%s]
            err[:] += dt*(b[j] - bhat[j])*fY_hat[(j+offset[0])%s]

        est = 0.0
        sc[:] = aTOL + np.maximum(np.abs(U_hat),np.abs(U_hat_new))*rTOL
        if errnorm == "2":
            est_to_bcast = None
            nsquared = np.zeros(U.shape[0],dtype=U.dtype)
            for k in range(U.shape[0]):
                nsquared[k] = FFT.comm.reduce(np.sum(np.power(np.abs(err[k]/sc[k]),2)))
            if FFT.comm.rank == 0:
                est_to_bcast = np.zeros(1,dtype=U.dtype)
                est = np.max(np.sqrt(nsquared))
                if context.dim == 3:
                    est /= np.sqrt(N[0]*N[1]*(N[2]/2 + 1))
                else:
                    est /= np.sqrt(N[0]*(N[1]/2+1))
                est_to_bcast[0] = est
            est_to_bcast = FFT.comm.bcast(est_to_bcast,root=0)
            est = est_to_bcast[0]
        elif errnorm == "inf":
            raise AssertionError("Don't use this, not sure if it works")
            #TODO: Test this error norm
            sc[:] = aTOL + np.maximum(np.abs(U_hat),np.abs(U_hat_new))*rTOL
            err[:] = err[:]/sc[:]
            err = np.abs(err,out=err)
            asdf = np.max(err)
            x = np.zeros(asdf.shape,U.dtype)
            FFT.comm.Allreduce(asdf,x,op=MPI.MAX)
            est = np.abs(np.max(x))
            est /= np.sqrt(N[0]*N[1]*(N[2]/2 + 1))
        else:
           assert False,"Wrong error norm"

        #Check error estimate
        exponent = 1.0/(err_order + 1)
        if not predictivecontroller:
            factor = min(facmax,max(facmin,fac*pow((1/est),exponent)))
        else:
            if not "last_dt" in context.time_integrator:
                context.time_integrator["last_dt"] = dt
            if not "last_est" in context.time_integrator:
                context.time_integrator["last_est"] = est

            last_dt = context.time_integrator["last_dt"]
            last_est = context.time_integrator["last_est"]
            factor = min(facmax,max(facmin,fac*pow((1/est),exponent)*dt/last_dt*pow(last_est/est,exponent)))
        if adaptive:
            dt = dt*factor
            if  est > 1.0:
                facmax = 1
                kw["additional_callback"](is_step_rejected_callback=True,dt_rejected=dt_prev,**kw)
                #The offset gets decreased in the  next step, which is something we do not want.
                if fsal:
                    offset[0] += 1
                continue

        if predictivecontroller:
            context.time_integrator["last_dt"] = dt_prev
            context.time_integrator["last_est"] = est
        break


    #Update U_hat and U
    U_hat[:] = U_hat_new
    return U_hat,dt,dt_prev
   
@optimizer
def getBS5(context,dU,ComputeRHS,aTOL,rTOL,adaptive=True,predictivecontroller=False):
    if not (context.solver_name in ["Bq2D","Bq3D"]):
        U = context.mesh_vars["U"]
        U_hat = context.mesh_vars["U_hat"]
    else:
        U = context.mesh_vars["Ur"]
        U_hat = context.mesh_vars["Ur_hat"]

    A = nodepy.rk.loadRKM("BS5").A.astype(np.float64)
    b = nodepy.rk.loadRKM("BS5").b.astype(np.float64)
    bhat = nodepy.rk.loadRKM("BS5").bhat.astype(np.float64)
    err_order = 4
    errnorm = "2"
    fsal = True

    #Offset for fsal stuff. #TODO: infer this from tstep
    offset = [0]

    s = A.shape[0]
    U_tmp = np.zeros(U.shape, dtype=U.dtype)
    fY_hat = np.zeros((s,) + U_hat.shape, dtype = U_hat.dtype)
    sc = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    err = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    U_hat_new = np.zeros(U_hat.shape,dtype=U_hat.dtype)

    #@wraps(adaptiveRK)
    def BS5(t,tstep,dt,additional_args = {}):
        return adaptiveRK(context,A,b,bhat,err_order, fY_hat,U_tmp,U_hat_new,sc,err, fsal,offset, aTOL,rTOL,adaptive,errnorm,dU,U_hat,ComputeRHS,dt,tstep,additional_args,predictivecontroller=predictivecontroller)
    return BS5

@optimizer
def getDP5(context,dU,ComputeRHS,aTOL,rTOL,adaptive=True,predictivecontroller=False):
    if not (context.solver_name in ["Bq2D","Bq3D"]):
        U = context.mesh_vars["U"]
        U_hat = context.mesh_vars["U_hat"]
    else:
        U = context.mesh_vars["Ur"]
        U_hat = context.mesh_vars["Ur_hat"]

    A = nodepy.rk.loadRKM("DP5").A.astype(np.float64)
    b = nodepy.rk.loadRKM("DP5").b.astype(np.float64)
    bhat = nodepy.rk.loadRKM("DP5").bhat.astype(np.float64)
    err_order = 4
    errnorm = "2"
    fsal = True

    #Offset for fsal stuff. #TODO: infer this from tstep
    offset = [0]

    s = A.shape[0]
    U_tmp = np.zeros(U.shape, dtype=U.dtype)
    fY_hat = np.zeros((s,) + U_hat.shape, dtype = U_hat.dtype)
    sc = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    err = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    U_hat_new = np.zeros(U_hat.shape,dtype=U_hat.dtype)

    #@wraps(adaptiveRK)
    def DP5(t,tstep,dt,additional_args = {}):
        return adaptiveRK(context,A,b,bhat,err_order, fY_hat,U_tmp,U_hat_new,sc,err, fsal,offset, aTOL,rTOL,adaptive,errnorm,dU,U_hat,ComputeRHS,dt,tstep,additional_args,predictivecontroller=predictivecontroller)
    return DP5

@optimizer
def getKCL5(context,dU,ComputeRHS,aTOL,rTOL,adaptive=True,predictivecontroller=False):
    if not (context.solver_name in ["Bq2D","Bq3D"]):
        U = context.mesh_vars["U"]
        U_hat = context.mesh_vars["U_hat"]
    else:
        U = context.mesh_vars["Ur"]
        U_hat = context.mesh_vars["Ur_hat"]

    A = np.zeros((8,8),dtype=np.float64)
    b = np.zeros(8,dtype=np.float64)
    bhat = np.zeros(8,dtype=np.float64)

    A[1,0] = 967290102210./6283494269639.
    A[2,1] = 852959821520./5603806251467.
    A[3,2] = 8043261511347./8583649637008
    A[4,3]= -115941139189./8015933834062.
    A[5,4] = 2151445634296./7749920058933.
    A[6,5] = 15619711431787./74684159414562.
    A[7,6] = 12444295717883./11188327299274.
    A[2,0] = 475331134681./7396070923784.
    A[3,1] = -8677837986029./16519245648862.
    A[4,2] = 2224500752467./10812521810777.
    A[5,3] = 1245361422071./3717287139065. 
    A[6,4] = 1652079198131./3788458824028.
    A[7,5] = -5225103653628./8584162722535.
    b[0] = 83759458317./1018970565139.
    b[1] = 0
    b[2] = 0
    b[3] = 0
    b[4] = 6968891091250./16855527649349.
    b[5] = 783521911849./8570887289572.
    b[6] = 3686104854613./11232032898210.
    b[7] = 517396786175./6104475356879.
    bhat[0] = -2632078767757./9365288548818.
    bhat[1] = 0
    bhat[2] = 138832778584802./30360463697573.
    bhat[3] = 7424139574315./5603229049946.
    bhat[4] = -32993229351515./6883415042289.
    bhat[5] = -3927384735361./7982454543710.
    bhat[6] = 9224293159931./15708162311543.
    bhat[7] = 624338737541./7691046757191.


    err_order = 4
    errnorm = "2"
    fsal = True

    #Offset for fsal stuff. #TODO: infer this from tstep
    offset = [0]

    s = A.shape[0]
    U_tmp = np.zeros(U.shape, dtype=U.dtype)
    fY_hat = np.zeros((s,) + U_hat.shape, dtype = U_hat.dtype)
    sc = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    err = np.zeros(U_hat.shape,dtype=U_hat.dtype)
    U_hat_new = np.zeros(U_hat.shape,dtype=U_hat.dtype)

    #@wraps(adaptiveRK)
    def KCL5(t,tstep,dt,additional_args = {}):
        return adaptiveRK(context,A,b,bhat,err_order, fY_hat,U_tmp,U_hat_new,sc,err, fsal,offset, aTOL,rTOL,adaptive,errnorm,dU,U_hat,ComputeRHS,dt,tstep,additional_args,predictivecontroller=predictivecontroller)
    return KCL5



@optimizer
def RK4(context,u0, u1, u2, dU, a, b, dt, ComputeRHS,kw):
    """Runge Kutta fourth order"""
    U = context.mesh_vars["U"]
    u2[:] = u1[:] = u0
    for rk in range(4):
        dU = ComputeRHS(context,U,u0,dU, rk)
        if rk == 0 and "additional_callback" in kw:
            kw["additional_callback"](fU_hat=dU,**kw)
        if rk < 3:
            u0[:] = u1 + b[rk]*dt*dU
        u2 += a[rk]*dt*dU
    u0[:] = u2
    return u0,dt,dt

@optimizer
def ForwardEuler(context,u0, u1, dU, dt, ComputeRHS,kw):
    U = context.mesh_vars["U"]
    dU = ComputeRHS(context,U,u0,dU, 0)        
    if "additional_callback" in kw:
        kw["additional_callback"](fU_hat=dU,**kw)
    u0 += dU*dt
    return u0,dt,dt

#TODO: Check whether we are really only doing forward euler at first and
#last step
@optimizer
def AB2(context,u0, u1,multistep_dt, dU, dt, tstep, ComputeRHS,kw):
    U = context.mesh_vars["U"]
    dU = ComputeRHS(context,U,u0,dU, 0)
    if "additional_callback" in kw:
        kw["additional_callback"](fU_hat=dU,**kw)
    if tstep == 0 or multistep_dt[0] != dt:
        multistep_dt[0] = dt
        u0 += dU*dt
    else:
        u0 += (1.5*dU*dt - 0.5*u1)        
    u1[:] = dU*dt    
    return u0,dt,dt

def getintegrator(context,ComputeRHS,f=None,g=None,ginv=None,gexp=None,hphi=None):
    if not (context.solver_name in ["Bq2D","Bq3D"]):
        dU = context.mesh_vars["dU"]
    else:
        dU = context.mesh_vars["dUr"]

    float = context.types["float"]
    """Return integrator using choice in global parameter integrator.
    """
    if context.solver_name in ("NS", "VV", "NS2D"):
        u0 = context.mesh_vars['U_hat']
    elif context.solver_name == "MHD":
        u0 = context.mesh_vars['UB_hat']
    elif context.solver_name in ["Bq2D","Bq3D"]:
        u0 = context.mesh_vars['Ur_hat']
    else:
        raise AssertionError("Not implemented")
    u1 = u0.copy()    

    if context.time_integrator["time_integrator_name"] == "RK4": 
        # RK4 parameters
        a = array([1./6., 1./3., 1./3., 1./6.], dtype=float)
        b = array([0.5, 0.5, 1.], dtype=float)
        u2 = u0.copy()
        @wraps(RK4)
        def func(t, tstep, dt,additional_args = {}):
            return RK4(context,u0, u1, u2, dU, a, b, dt, ComputeRHS,additional_args)
        return func
    elif context.time_integrator["time_integrator_name"] == "ForwardEuler":  
        @wraps(ForwardEuler)
        def func(t, tstep, dt,additional_args = {}):
            return ForwardEuler(context,u0, u1, dU, dt, ComputeRHS,additional_args)
        return func
    elif context.time_integrator["time_integrator_name"] == "BS5_adaptive": 
        TOL = context.time_integrator["TOL"]
        return getBS5(context,dU,ComputeRHS,aTOL=TOL,rTOL=TOL,adaptive=True)
    elif context.time_integrator["time_integrator_name"] == "DP5_adaptive":
        TOL = context.time_integrator["TOL"]
        return getDP5(context,dU,ComputeRHS,aTOL=TOL,rTOL=TOL,adaptive=True)
    elif context.time_integrator["time_integrator_name"] == "KCL5_adaptive":
        TOL = context.time_integrator["TOL"]
        return getKCL5(context,dU,ComputeRHS,aTOL=TOL,rTOL=TOL,adaptive=True)
    elif context.time_integrator["time_integrator_name"] == "BS5_adaptive_p": 
        TOL = context.time_integrator["TOL"]
        return getBS5(context,dU,ComputeRHS,aTOL=TOL,rTOL=TOL,adaptive=True,predictivecontroller=True)
    elif context.time_integrator["time_integrator_name"] == "BS5_fixed":
        TOL = 100 #This shouldn't get used
        return getBS5(context,dU,ComputeRHS,aTOL=TOL,rTOL=TOL,adaptive=False)
    elif context.time_integrator["time_integrator_name"] == "DP5_fixed":
        TOL = 100 #This shouldn't get used
        return getDP5(context,dU,ComputeRHS,aTOL=TOL,rTOL=TOL,adaptive=False)
    elif context.time_integrator["time_integrator_name"] == "KCL5_fixed":
        TOL = 100 #This shouldn't get used
        return getKCL5(context,dU,ComputeRHS,aTOL=TOL,rTOL=TOL,adaptive=False)
    elif context.time_integrator["time_integrator_name"] == "AB2":
        multistep_dt = [-1]
        @wraps(AB2)
        def func(t, tstep, dt,additional_args = {}):
            return AB2(context,u0, u1,multistep_dt, dU, dt, tstep, ComputeRHS,additional_args)
        return func
    elif context.time_integrator["time_integrator_name"] == "IMEX1":
        return getIMEX1(context,dU,f,g,ginv)
    elif context.time_integrator["time_integrator_name"] == "IMEX4":
        return getIMEX4(context,dU,f,g,ginv)
    elif context.time_integrator["time_integrator_name"] == "IMEX3":
        return getIMEX3(context,dU,f,g,ginv)
    elif context.time_integrator["time_integrator_name"] == "IMEX5":
        return getIMEX5(context,dU,f,g,ginv)
    elif context.time_integrator["time_integrator_name"] == "EXPBS5":
        return getexpBS5(context,dU,f,gexp)
    elif context.time_integrator["time_integrator_name"] == "EXPEULER":
        def func(t,tstep,dt,additional_args = {}):
            return expEuler(context,dU,f,gexp,hphi,t,tstep,dt,additional_args)
        return func
    else:
        raise AssertionError("Please specifiy a  time integrator")
