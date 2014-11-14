import pyfftw

nthreads = 1
# Keep fft objects in cache for efficiency
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(1e8)
def empty(N, dtype="float", bytes=16):
    return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)

# Monkey patches for fft
#ifft = pyfftw.interfaces.numpy_fft.ifft
#fft = pyfftw.interfaces.numpy_fft.fft
#fft2 = pyfftw.interfaces.numpy_fft.fft2
#ifft2 = pyfftw.interfaces.numpy_fft.ifft2

def fft2(inarray, axes=(-2,-1)):
    return pyfftw.interfaces.numpy_fft.fft2(inarray, 
                                        axes=axes,
                                        planner_effort="FFTW_MEASURE",
                                        threads=nthreads)

def ifft2(inarray, axes=(-2,-1)):
    return pyfftw.interfaces.numpy_fft.ifft2(inarray, 
                                            axes=axes,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)

def fft(inarray, axis=-1):
    return pyfftw.interfaces.numpy_fft.fft(inarray, 
                                            axis=axis,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)

def ifft(inarray, axis=-1):
    return pyfftw.interfaces.numpy_fft.ifft(inarray, 
                                            axis=axis,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)

def ifftn(inarray, axes=(-3, -2,-1)):
    return pyfftw.interfaces.numpy_fft.ifftn(inarray, 
                                            axes=axes,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)

def fftn(inarray, axes=(-3, -2,-1)):
    return pyfftw.interfaces.numpy_fft.fftn(inarray, 
                                            axes=axes,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)

def rfft2(inarray, axes=(-2,-1)):
    return pyfftw.interfaces.numpy_fft.rfft2(inarray, 
                                        axes=axes,
                                        planner_effort="FFTW_MEASURE",
                                        threads=nthreads)

def irfft2(inarray, axes=(-2,-1)):
    return pyfftw.interfaces.numpy_fft.irfft2(inarray, 
                                            axes=axes,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)
        
def rfft(inarray, axis=-1):
    return pyfftw.interfaces.numpy_fft.rfft(inarray, 
                                            axis=axis,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)

def irfft(inarray, axis=-1):
    return pyfftw.interfaces.numpy_fft.irfft(inarray, 
                                            axis=axis,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)

def rfftn(inarray, axes=(-3, -2, -1), s=None):
    return pyfftw.interfaces.numpy_fft.rfftn(inarray, 
                                            axes=axes,
                                            s=s,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)

def irfftn(inarray, axes=(-3, -2, -1), s=None):
    return pyfftw.interfaces.numpy_fft.irfftn(inarray, 
                                            axes=axes,
                                            planner_effort="FFTW_MEASURE",
                                            threads=nthreads)
