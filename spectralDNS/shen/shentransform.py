from collections import defaultdict
from mpiFFT4py import dct, work_arrays, Slab_R2C, fftfreq, rfft2, \
    irfft2, rfft, irfft, fft, ifft
from mpi4py import MPI
from numpy import array, zeros, zeros_like, sum, hstack, meshgrid, abs, \
    pi, uint8, rollaxis, arange
import numpy as np
from ..optimization import optimizer

work = work_arrays()


class SlabShen_R2C(Slab_R2C):

    def __init__(self, N, L, comm, padsize=1.5, threads=1, communication='Alltoall', dealias_cheb=False,
                 planner_effort=defaultdict(lambda: "FFTW_MEASURE", {"dct": "FFTW_EXHAUSTIVE"})):
        Slab_R2C.__init__(self, N, L, comm, "double", padsize=padsize, threads=threads,
                          communication=communication, planner_effort=planner_effort)
        self.dealias_cheb = dealias_cheb

    def complex_shape_padded_T(self):
        """The local shape of the transposed complex data padded in x and z directions"""
        if self.dealias_cheb:
            return (int(self.padsize*self.Np[0]), int(self.padsize*self.N[1]), int(self.padsize*self.N[2]/2+1))
        else:
            return (self.Np[0], int(self.padsize*self.N[1]), int(self.padsize*self.N[2]/2+1))

    def real_shape_padded(self):
        """The local shape of the real data"""
        if self.dealias_cheb:
            return (int(self.padsize*self.Np[0]), int(self.padsize*self.N[1]), int(self.padsize*self.N[2]))
        else:
            return (self.Np[0], int(self.padsize*self.N[1]), int(self.padsize*self.N[2]))

    def complex_shape_padded(self):
        if self.dealias_cheb:
            return (int(self.padsize*self.N[0]), int(self.padsize*self.Np[1]), int(self.padsize*self.N[2]/2+1))
        else:
            return (self.N[0], int(self.padsize*self.Np[1]), int(self.padsize*self.N[2]/2+1))

    def get_mesh_dims(self, ST):
        return [self.get_mesh_dim(ST, i) for i in range(3)]

    def real_local_slice(self, padsize=1):
        if self.dealias_cheb:
            return (slice(self.rank*padsize*self.Np[0], (self.rank+1)*padsize*self.Np[0], 1),
                    slice(0, int(padsize*self.N[1]), 1),
                    slice(0, int(padsize*self.N[2]), 1))
        else:
            return (slice(self.rank*self.Np[0], (self.rank+1)*self.Np[0], 1),
                    slice(0, int(padsize*self.N[1]), 1),
                    slice(0, int(padsize*self.N[2]), 1))

    def global_complex_shape(self, padsize=1.0):
        """Global size of problem in complex wavenumber space"""
        if self.dealias_cheb:
            return (int(self.padsize*self.N[0]), int(self.padsize*self.N[1]),
                    int(padsize*self.N[2]/2+1))
        else:
            return (self.N[0], int(self.padsize*self.N[1]),
                    int(padsize*self.N[2]/2+1))

    def get_mesh_dim(self, ST, d):
        if d == 0:
            return ST.points_and_weights(self.N[0], ST.quad)[0]
        elif d == 1:
            return arange(self.N[1], dtype=self.float)*self.L[1]/self.N[1]
        elif d == 2:
            return arange(self.N[2], dtype=self.float)*self.L[2]/self.N[2]

    def get_local_mesh(self, ST):
        x0, x1, x2 = self.get_mesh_dims(ST)

        # Get grid for velocity points
        X = meshgrid(x0[int(self.rank*self.Np[0]):int((self.rank+1)*self.Np[0])],
                     x1, x2, indexing='ij', sparse=True)
        X = [np.broadcast_to(x, self.real_shape()) for x in X]
        return X

    def get_local_wavenumbermesh(self, scaled=False):
        """Returns (scaled) local decomposed wavenumbermesh

        If scaled is True, then the wavenumbermesh is scaled with physical mesh
        size. This takes care of mapping the physical domain to a computational
        cube of size (2pi)**3
        """
        kx = arange(self.N[0]).astype(self.float)
        ky = fftfreq(self.N[1], 1./self.N[1])[int(self.rank*self.Np[1]):int((self.rank+1)*self.Np[1])]
        kz = fftfreq(self.N[2], 1./self.N[2])[:self.Nf]
        kz[-1] *= -1.0
        Ks = meshgrid(kx, ky, kz, indexing='ij', sparse=True)
        if scaled:
            Lp = array([2, 2*pi, 2*pi])/self.L
            for i in range(3):
                Ks[i] *= Lp[i]
        K = [np.broadcast_to(k, self.complex_shape()) for k in Ks]
        return K

    def get_scaled_local_wavenumbermesh(self):
        K = self.get_local_wavenumbermesh()

        # scale with physical mesh size.
        # This takes care of mapping the physical domain to a computational cube of size (2, 2pi, 2pi)
        # Note that first direction cannot be different from 2 (yet)
        Lp = array([2, 2*pi, 2*pi])/self.L
        for i in range(3):
            K[i] *= Lp[i]
        return K

    def get_dealias_filter(self):
        """Filter for dealiasing nonlinear convection"""
        K = self.get_local_wavenumbermesh()
        kmax = 2./3.*(self.N/2)
        kmax[0] = 2./3*self.N[0] if self.dealias_cheb else self.N[0]
        dealias = array((abs(K[0]) < kmax[0])*(abs(K[1]) < kmax[1])*
                        (abs(K[2]) < kmax[2]), dtype=uint8)
        return dealias

    @staticmethod
    @optimizer
    def copy_from_padded(fp, fu, N, axis=0):
        if axis == 1:
            fu.fill(0)
            fu[:, :N[1]//2+1] = fp[:, :N[1]//2+1, :(N[2]//2+1)]
            fu[:, N[1]//2:] += fp[:, -N[1]//2:, :(N[2]//2+1)]
        elif axis == 2:
            fu[:] = fp[:, :, :(N[2]//2+1)]

        return fu

    @staticmethod
    @optimizer
    def copy_to_padded(fu, fp, N, axis=0):
        if axis == 0:
            fp[:N[0]] = fu[:N[0]]
        elif axis == 1:
            fp[:, :N[1]//2] = fu[:, :N[1]//2]
            fp[:, -N[1]//2:] = fu[:, N[1]//2:]
        elif axis == 2:
            fp[:, :, :(N[2]//2+1)] = fu[:]
        return fp

    def _forward(self, u, fu, fun, dealias=None):

        # Intermediate work arrays
        Uc_hat = self.work_arrays[(self.complex_shape(), self.complex, 0, False)]

        if self.num_processes == 1:

            if not dealias == '3/2-rule':
                assert u.shape == self.real_shape()

                Uc_hat = rfft2(u, Uc_hat, axes=(1, 2), threads=self.threads,
                               planner_effort=self.planner_effort['rfft2'])
                fu = fun(Uc_hat, fu)

            else:
                if not self.dealias_cheb:
                    Upad_hat = self.work_arrays[(self.complex_shape_padded(), self.complex, 0, False)]
                    Upad_hat_z = self.work_arrays[((self.N[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0, False)]

                    Upad_hat = rfft(u, Upad_hat, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                    Upad_hat_z = SlabShen_R2C.copy_from_padded(Upad_hat, Upad_hat_z, self.N, 2)
                    Upad_hat_z[:] = fft(Upad_hat_z, axis=1, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['fft'])
                    Uc_hat = SlabShen_R2C.copy_from_padded(Upad_hat_z, Uc_hat, self.N, 1)
                    fu = fun(Uc_hat/self.padsize**2, fu)
                else:
                    # Intermediate work arrays required for transform
                    Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                    Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1, False)]
                    Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0, False)]
                    Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]

                    # Do ffts and truncation in the padded y and z directions
                    Upad_hat3 = rfft(u, Upad_hat3, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                    Upad_hat2 = SlabShen_R2C.copy_from_padded(Upad_hat3, Upad_hat2, self.N, 2)
                    Upad_hat2[:] = fft(Upad_hat2, axis=1, threads=self.threads, planner_effort=self.planner_effort['fft'])
                    Upad_hat = SlabShen_R2C.copy_from_padded(Upad_hat2, Upad_hat, self.N, 1)

                    # Perform fst of data in x-direction
                    Upad_hat0 = fun(Upad_hat, Upad_hat0)

                    # Truncate to original complex shape
                    fu[:] = Upad_hat0[:self.N[0]]/self.padsize**2
            return fu

        if not dealias == '3/2-rule':

            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
            Uc_hat = self.work_arrays[(fu, 0, False)]

            if self.communication == 'Alltoall':
                #Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                #Uc_hatT = rfft2(u, Uc_hatT, axes=(1,2), threads=self.threads, planner_effort=self.planner_effort['rfft2'])
                #Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                #self.comm.Alltoall(MPI.IN_PLACE, [Uc_hat, self.mpitype])

                # Intermediate work array required for transform
                U_mpi = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]

                # Do 2 ffts in y-z directions on owned data
                Uc_hatT = rfft2(u, Uc_hatT, axes=(1, 2), threads=self.threads, planner_effort=self.planner_effort['rfft2'])

                #Transform data to align with x-direction
                U_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)

                #Communicate all values
                self.comm.Alltoall([U_mpi, self.mpitype], [Uc_hat, self.mpitype])


            elif self.communication == 'Alltoallw':
                if len(self._subarraysA) == 0:
                    self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()

                # Do 2 ffts in y-z directions on owned data
                Uc_hatT = rfft2(u, Uc_hatT, axes=(1, 2), threads=self.threads,
                                planner_effort=self.planner_effort['rfft2'])

                self.comm.Alltoallw(
                    [Uc_hatT, self._counts_displs, self._subarraysB],
                    [Uc_hat,  self._counts_displs, self._subarraysA])

            fu = fun(Uc_hat, fu)

        else:
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]

            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0, False)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0, False)]

                Upad_hatT = rfft(u, Upad_hatT, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                Upad_hat_z = SlabShen_R2C.copy_from_padded(Upad_hatT, Upad_hat_z, self.N, 2)
                Upad_hat_z[:] = fft(Upad_hat_z, axis=1, threads=self.threads, planner_effort=self.planner_effort['fft'])
                Uc_hatT = SlabShen_R2C.copy_from_padded(Upad_hat_z, Uc_hatT, self.N, 1)

                if self.communication == 'Alltoall':
                    #Uc_mpi  = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
                    #Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                    #self.comm.Alltoall(MPI.IN_PLACE, [Uc_hat, self.mpitype])

                    Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 2, False)]
                    Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
                    self.comm.Alltoall([Uc_mpi, self.mpitype], [Uc_hat, self.mpitype])

                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA) == 0:
                        self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()
                    self.comm.Alltoallw(
                        [Uc_hatT, self._counts_displs, self._subarraysB],
                        [Uc_hat,  self._counts_displs, self._subarraysA])

                fu = fun(Uc_hat/self.padsize**2, fu)

            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"
                assert u.shape == self.real_shape_padded()

                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1, False)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0, False)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0, False)]

                # Do ffts and truncation in the padded y and z directions
                Upad_hat3 = rfft(u, Upad_hat3, axis=2, threads=self.threads, planner_effort=self.planner_effort['rfft'])
                Upad_hat2 = SlabShen_R2C.copy_from_padded(Upad_hat3, Upad_hat2, self.N, 2)
                Upad_hat2[:] = fft(Upad_hat2, axis=1, threads=self.threads, planner_effort=self.planner_effort['fft'])
                Upad_hat1 = SlabShen_R2C.copy_from_padded(Upad_hat2, Upad_hat1, self.N, 1)

                if self.communication == 'Alltoall':
                    # Transpose and commuincate data
                    U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                    U_mpi[:] = rollaxis(Upad_hat1.reshape(self.complex_shape_padded_I()), 1)
                    self.comm.Alltoall(MPI.IN_PLACE, [Upad_hat, self.mpitype])

                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA_pad) == 0:
                        self._subarraysA_pad, self._subarraysB_pad, self._counts_displs = self.get_subarrays(padsize=self.padsize)

                    self.comm.Alltoallw(
                        [Upad_hat1, self._counts_displs, self._subarraysB_pad],
                        [Upad_hat,  self._counts_displs, self._subarraysA_pad])

                # Perform fst of data in x-direction
                Upad_hat0 = fun(Upad_hat, Upad_hat0)

                # Truncate to original complex shape
                fu[:] = Upad_hat0[:self.N[0]]/self.padsize**2

        return fu

    def backward(self, fu, u, fun, dealias=None):

        Uc_hat = self.work_arrays[(self.complex_shape(), self.complex, 0, False)]
        Uc_mpi = Uc_hat.reshape((self.num_processes, self.Np[0], self.Np[1], self.Nf))
        fun = fun.backward

        if dealias == '2/3-rule' and self.dealias.shape == (0,):
            self.dealias = self.get_dealias_filter()

        if self.num_processes == 1:
            if not dealias == '3/2-rule':
                if dealias == '2/3-rule':
                    fu *= self.dealias

                Uc_hat = fun(fu, Uc_hat)
                u = irfft2(Uc_hat, u, axes=(1, 2), overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft2'])

            else:
                if not self.dealias_cheb:
                    Upad_hat = self.work_arrays[(self.complex_shape_padded(), self.complex, 0)]
                    Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0)]

                    Uc_hat = fun(fu*self.padsize**2, Uc_hat)
                    Upad_hat_z = SlabShen_R2C.copy_to_padded(Uc_hat, Upad_hat_z, self.N, 1)
                    Upad_hat_z[:] = ifft(Upad_hat_z, axis=1, threads=self.threads, planner_effort=self.planner_effort['ifft'])
                    Upad_hat = SlabShen_R2C.copy_to_padded(Upad_hat_z, Upad_hat, self.N, 2)
                    u = irfft(Upad_hat, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])

                else:
                    # Intermediate work arrays required for transform
                    Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                    Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                    Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
                    Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                    Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]

                    # Expand in x-direction and perform ifst
                    Upad_hat0 = SlabShen_R2C.copy_to_padded(fu*self.padsize**2, Upad_hat0, self.N, 0)
                    Upad_hat = fun(Upad_hat0, Upad_hat)

                    Upad_hat2 = SlabShen_R2C.copy_to_padded(Upad_hat, Upad_hat2, self.N, 1)
                    Upad_hat2[:] = ifft(Upad_hat2, axis=1, threads=self.threads, planner_effort=self.planner_effort['ifft'])

                    # pad in z-direction and perform final irfft
                    Upad_hat3 = SlabShen_R2C.copy_to_padded(Upad_hat2, Upad_hat3, self.N, 2)
                    u = irfft(Upad_hat3, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])

            return u

        if not dealias == '3/2-rule':
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
            if dealias == '2/3-rule':
                fu *= self.dealias

            Uc_hat = fun(fu, Uc_hat)

            if self.communication == 'Alltoall':
                self.comm.Alltoall(MPI.IN_PLACE, [Uc_hat, self.mpitype])
                Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
                #Uc_mpi  = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]
                #self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])
                #Uc_hatT = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())

            elif self.communication == 'Alltoallw':
                if len(self._subarraysA) == 0:
                    self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()
                Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
                self.comm.Alltoallw(
                    [Uc_hat, self._counts_displs, self._subarraysA],
                    [Uc_hatT,  self._counts_displs, self._subarraysB])

            u = irfft2(Uc_hatT, u, axes=(1, 2), overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft2'])

        else:
            Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0, False)]
            if not self.dealias_cheb:
                Upad_hatT = self.work_arrays[(self.complex_shape_padded_T(), self.complex, 0)]
                Upad_hat_z = self.work_arrays[((self.Np[0], int(self.padsize*self.N[1]), self.Nf), self.complex, 0)]

                Uc_hat = fun(fu*self.padsize**2, Uc_hat)
                if self.communication == 'Alltoall':
                    # In-place
                    #self.comm.Alltoall(MPI.IN_PLACE, [Uc_hat, self.mpitype])
                    # Not in-place
                    Uc_mpi = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0, False)]
                    self.comm.Alltoall([Uc_hat, self.mpitype], [Uc_mpi, self.mpitype])

                    Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())

                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA) == 0:
                        self._subarraysA, self._subarraysB, self._counts_displs = self.get_subarrays()

                    self.comm.Alltoallw(
                        [Uc_hat, self._counts_displs, self._subarraysA],
                        [Uc_hatT, self._counts_displs, self._subarraysB])

                Upad_hat_z = SlabShen_R2C.copy_to_padded(Uc_hatT, Upad_hat_z, self.N, 1)
                Upad_hat_z[:] = ifft(Upad_hat_z, axis=1, threads=self.threads, planner_effort=self.planner_effort['ifft'])
                Upad_hatT = SlabShen_R2C.copy_to_padded(Upad_hat_z, Upad_hatT, self.N, 2)
                u = irfft(Upad_hatT, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])

            else:
                assert self.num_processes <= self.N[0]/2, "Number of processors cannot be larger than N[0]/2 for 3/2-rule"

                # Intermediate work arrays required for transform
                Upad_hat  = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 0, False)]
                Upad_hat0 = self.work_arrays[(self.complex_shape_padded_0(), self.complex, 1)]
                Upad_hat1 = self.work_arrays[(self.complex_shape_padded_1(), self.complex, 0, False)]
                Upad_hat2 = self.work_arrays[(self.complex_shape_padded_2(), self.complex, 0)]
                Upad_hat3 = self.work_arrays[(self.complex_shape_padded_3(), self.complex, 0)]

                # Expand in x-direction and perform ifst
                Upad_hat0 = SlabShen_R2C.copy_to_padded(fu*self.padsize**2, Upad_hat0, self.N, 0)
                Upad_hat = fun(Upad_hat0, Upad_hat)

                if self.communication == 'Alltoall':
                    # Communicate to distribute first dimension (like Fig. 2b but padded in x-dir)
                    self.comm.Alltoall(MPI.IN_PLACE, [Upad_hat, self.mpitype])

                    # Transpose data and pad in y-direction before doing ifft. Now data is padded in x and y
                    U_mpi = Upad_hat.reshape(self.complex_shape_padded_0_I())
                    Upad_hat1[:] = rollaxis(U_mpi, 1).reshape(Upad_hat1.shape)

                elif self.communication == 'Alltoallw':
                    if len(self._subarraysA_pad) == 0:
                        self._subarraysA_pad, self._subarraysB_pad, self._counts_displs = self.get_subarrays(padsize=self.padsize)
                    self.comm.Alltoallw(
                        [Upad_hat,  self._counts_displs, self._subarraysA_pad],
                        [Upad_hat1, self._counts_displs, self._subarraysB_pad])

                Upad_hat2 = SlabShen_R2C.copy_to_padded(Upad_hat1, Upad_hat2, self.N, 1)
                Upad_hat2[:] = ifft(Upad_hat2, axis=1, threads=self.threads, planner_effort=self.planner_effort['ifft'])

                # pad in z-direction and perform final irfft
                Upad_hat3 = SlabShen_R2C.copy_to_padded(Upad_hat2, Upad_hat3, self.N, 2)
                u = irfft(Upad_hat3, u, axis=2, overwrite_input=True, threads=self.threads, planner_effort=self.planner_effort['irfft'])

        return u

    def forward(self, u, fu, S, dealias=None):
        """Fast Shen transform of x-direction, Fourier transform of y and z"""
        fu = self._forward(u, fu, S.forward, dealias=dealias)
        return fu

    def scalar_product(self, u, fu, S, dealias=None):
        """Fast Shen scalar product of x-direction, Fourier transform of y and z"""
        fu = self._forward(u, fu, S.scalar_product, dealias=dealias)
        return fu

    def fft(self, u, fu):
        """Fast Fourier transform of y and z"""
        # Intermediate work arrays
        Uc_mpi = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        Uc_hatT = rfft2(u, Uc_hatT, axes=(1, 2), threads=self.threads, planner_effort=self.planner_effort['rfft2'])
        Uc_mpi[:] = rollaxis(Uc_hatT.reshape(self.Np[0], self.num_processes, self.Np[1], self.Nf), 1)
        self.comm.Alltoall([Uc_mpi, self.mpitype], [fu, self.mpitype])
        return fu

    def ifft(self, fu, u):
        """Inverse Fourier transforms in y and z"""
        Uc_mpi = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.Nf), self.complex, 0)]
        Uc_hatT = self.work_arrays[(self.complex_shape_T(), self.complex, 0)]
        self.comm.Alltoall([fu, self.mpitype], [Uc_mpi, self.mpitype])
        Uc_hatT[:] = rollaxis(Uc_mpi, 1).reshape(self.complex_shape_T())
        u = irfft2(Uc_hatT, u, axes=(1, 2), threads=self.threads, planner_effort=self.planner_effort['irfft2'])
        return u

    def fct0(self, u, fu, S):
        """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do forward."""
        U_mpi2 = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.N[2]), self.float, 0)]
        UT = self.work_arrays[((self.N[0], self.Np[1], self.N[2]), self.float, 0)]
        U_mpi2[:] = rollaxis(u.reshape(self.Np[0], self.num_processes, self.Np[1], self.N[2]), 1)
        self.comm.Alltoall([U_mpi2, self.mpitype], [UT, self.mpitype])
        fu = S.forward(UT, fu)
        return fu

    def ifct0(self, fu, u, S):
        """Fast Cheb transform of x-direction. No FFT, just align data in x-direction and do ifct"""
        U_mpi2 = self.work_arrays[((self.num_processes, self.Np[0], self.Np[1], self.N[2]), self.float, 0)]
        UT = self.work_arrays[((self.N[0], self.Np[1], self.N[2]), self.float, 0)]
        UT = S.backward(fu, UT)
        self.comm.Alltoall([UT, self.mpitype], [U_mpi2, self.mpitype])
        u[:] = rollaxis(U_mpi2, 1).reshape(u.shape)
        return u


    def dx(self, u, quad):
        """Compute integral of u over domain"""
        uu = sum(u, axis=(1, 2))
        c = zeros(self.N[0])
        self.comm.Gather(uu, c)
        if self.rank == 0:
            if quad == 'GL':
                ak = zeros_like(c)
                ak = dct(c, ak, 1, axis=0)
                ak /= (self.N[0]-1)
                w = arange(0, self.N[0], 1, dtype=self.float)
                w[2:] = 2./(1-w[2:]**2)
                w[0] = 1
                w[1::2] = 0
                return sum(ak*w)*self.L[1]*self.L[2]/self.N[1]/self.N[2]

            elif quad == 'GC':
                d = zeros(self.N[0])
                k = 2*(1 + arange((self.N[0]-1)//2))
                d[::2] = (2./self.N[0])/hstack((1., 1.-k*k))
                w = zeros_like(d)
                w = dct(d, w, type=3, axis=0)
                return sum(c*w)*self.L[1]*self.L[2]/self.N[1]/self.N[2]
        else:
            return 0
