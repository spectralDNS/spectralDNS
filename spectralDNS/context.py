__author__ = "Nathanael Schilling <schillna@in.tum.de>"
__date__ = "2016-04-11"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""
This file defines a class for managing everything that was previously done using global variables.
Currently only works for the NS solver
"""

import spectralDNS.mesh.triplyperiodic
import spectralDNS.mesh.doublyperiodic
import spectralDNS.config
from mpi4py import MPI
import numpy as np
from mpiFFT4py import Slab_R2C, Pencil_R2C, Line_R2C, work_arrays

class Context:
    """
    This class contains all of the information necessary for a run
    mesh - The mesh type (e.g. triplyperiodic)
    comm - points to MPI.COM_WORLD
    rank - The MPI rank
    types - A dictionary with "float", "complex" and "mpitype" keys
    solver_name - Something like NS or MHD
    model_params - A dictionary with keys of "M","L","nu","dx","T"
    time_integrator - A dictionary with keys of time_integrator_name, dt, integrator
    profiler - Not sure what exactly this is, but is what was previously global variable
    make_profile - Whether to profile
    solver - The module of the solver
    callbacks - A dictionary with possible keys update, additional_callback, regression_test
    silent - Whether to output stuff to stdout
    threads - Number of threads to use
    """
    #TODO:Find out if we need to get precision here or can get it from argv.
    def __init__(self,mesh,argv,precision="double",silent=False):
        self.mesh = mesh
        self.silent = silent

        comm = self.comm = MPI.COMM_WORLD
        self.comm.barrier()
        self.num_processes = comm.Get_size()
        self.rank = comm.Get_rank()
        self.model_params = {}
        if mesh is "triplyperiodic":
            #Get the results from the command-line parsing routine, which may return things as strings.
            args = spectralDNS.config.triplyperiodic.parse_args(argv)
        elif mesh is "doublyperiodic":
            args = spectralDNS.config.doublyperiodic.parse_args(argv)
        else:
            raise AssertionError("Not yet implemented")

        precision = self.precision = args.precision
        self.threads = args.threads
        self.types = {}
        self.types["float"], self.types["complex"], self.types["mpitype"] = {"single": (np.float32, np.complex64, MPI.F_FLOAT_COMPLEX),
                                   "double": (np.float64, np.complex128, MPI.F_DOUBLE_COMPLEX)}[precision]
        M = self.model_params["M"] = np.array([eval(str(f)) for f in args.M], dtype=int)
        L = self.model_params["L"] = np.array([eval(str(f)) for f in args.L], dtype=self.types["float"])
        if(np.allclose(L,np.ones(len(M),dtype=self.types["float"])*2*np.pi)):
            L = self.model_params["L"] = np.ones(len(M),dtype=self.types["float"])*2*np.pi
        N = self.model_params["N"] = 2**M
        dx = self.model_params["dx"] = (L/N).astype(self.types["float"])
        nu = self.model_params["nu"] = self.types["float"](args.nu)
        T = self.model_params["T"] = self.types["float"](args.T)
        self.model_params["t"] = 0.0
        #TODO: Check that this is working
        self.dealias_name = args.dealias

        
        self.solver_name = args.solver
        self.decomposition = args.decomposition
        kwargs = {}
        if self.decomposition == "pencil":
            kwargs["P1"] = args.P1
            kwargs["alignment"] = args.Pencil_alignment

        self.initialize_mesh_variables(decomposition=args.decomposition,N=N,precision=args.precision,**kwargs)
        self.time_integrator = {}
        self.time_integrator["dt"] = self.types["float"](args.dt)
        self.time_integrator["time_integrator_name"] = args.integrator
        self.time_integrator["TOL"] = args.time_integrator_tol

        self.make_profile = args.make_profile
        if args.make_profile: self.profiler = cProfile.Profile()

        self.optimization = args.optimization
        self.communication = args.communication
        self.write_result = args.write_result
        self.write_yz_slice = args.write_yz_slice
        self.write_xz_slice = args.write_xz_slice
        self.write_xy_slice = args.write_xy_slice
        self.checkpoint = args.checkpoint


        if mesh is "triplyperiodic":

            if args.solver == 'NS':
                import spectralDNS.solvers.NS as solver
                
            elif args.solver == 'VV':
                import spectralDNS.solvers.VV as solver

            elif args.solver == 'MHD':
                import spectralDNS.solvers.MHD as solver
            elif args.solver == "Bq3D":
                import spectralDNS.solvers.NS_Boussinesq as solver
            else:
                raise AttributeError("Wrong solver!")

        elif mesh is "doublyperiodic":        
        
            if args.solver == 'NS2D':
                import spectralDNS.solvers.NS2D as solver

            elif args.solver == 'Bq2D':
                import spectralDNS.solvers.NS2D_Boussinesq as solver
                
            else:
                raise AttributeError("Wrong solver!")

        elif mesh is "channel":
            
            if args.solver == 'IPCS':
                import spectralDNS.solvers.ShenDNS as solver           
                
            elif args.solver == 'IPCSR':
                import spectralDNS.solvers.ShenDNSR as solver
                
            elif args.solver == 'KMM':
                import spectralDNS.solvers.ShenKMM as solver

            elif args.solver == 'KMMRK3':
                import spectralDNS.solvers.ShenKMMRK3 as solver
            
            else:
                raise AttributeError("Wrong solver!")

        #elif family is "ShenMHD":

            #if args.solver == 'IPCS_MHD':
                #import spectralDNS.solvers.ShenMHD as solver            
            
            #else:
                #raise AttributeError("Wrong solver!")

        #elif family is "ShenGeneralBCs":
            
            #if args.solver == 'IPCS_GeneralBCs':
                #import spectralDNS.solvers.ShenDNSGeneralBCs as solver
                
            #else:
                #raise AttributeError("Wrong solver!")
        solver.initializeContext(self,args)
        self.solver = solver
        self.callbacks = {}
        def default_update(t,dt,tstep,context):
            pass
        def default_additional_callback(**kwargs):
            pass
        def default_regression_test(t,tstep,context):
            pass


        
    def initialize_mesh_variables(self,decomposition,N,precision,**kwargs):
        if hasattr(self,"mesh_vars"):
           raise AttributeError("Only call this once for a given context") 
        if not (self.mesh in ["triplyperiodic","doublyperiodic"]):
            raise AssertionError("Won't work yet as spectralDNS/mesh/* files haven't all been updated yet")

        if self.mesh in ('doublyperiodic', 'triplyperiodic'):
            if decomposition == 'slab':
                FFT = Slab_R2C(self.model_params["N"], self.model_params["L"], MPI.COMM_WORLD, precision,threads=self.threads)
            elif decomposition == 'pencil':
                FFT = Pencil_R2C(self.model_params["N"], self.model_params["L"], MPI.COMM_WORLD, precision, P1=kwargs["P1"], alignment=kwargs["alignment"],threads=self.threads)
            elif decomposition == 'line':
                FFT = Line_R2C(self.model_params["N"], self.model_params["L"], MPI.COMM_WORLD, precision,threads=self.threads)

        self.MPI = MPI
        self.FFT = FFT

        if self.mesh == "triplyperiodic":
            self.dim = 3
            self.mesh_vars = spectralDNS.mesh.triplyperiodic.setup(self.solver_name,context=self)
            self.mesh_info = {"decomposition":decomposition,"N":N,precision:"precision"}
        elif self.mesh == "doublyperiodic":
            self.dim = 2
            self.mesh_vars = spectralDNS.mesh.doublyperiodic.setup(self.solver_name,context=self)
            self.FFT.fftn = self.FFT.fft2
            self.FFT.ifftn = self.FFT.ifft2

        if decomposition == 'pencil':
            self.mesh_info["P1"] = kwargs["P1"]
            self.mesh_info["alignment"] = kwargs["alignment"]
        self.work = work_arrays()
