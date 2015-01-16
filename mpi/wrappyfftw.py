__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-11-19"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

__all__ = ['fft', 'ifft', 'fft2', 'ifft2', 'fftn', 'ifftn',
           'rfft', 'irfft', 'rfft2', 'irfft2', 'rfftn', 'irfftn', 
           'fftfreq', 'empty', 'zeros']

from numpy import empty, zeros
from numpy.fft import fftfreq, fft, ifft, fftn, ifftn, rfft, irfft, rfft2, irfft2, rfftn, irfftn, fft2, ifft2

try:
    import pyfftw
    nzeros = zeros
    nthreads = 1
    def empty(N, dtype=float, bytes=16):
        return pyfftw.n_byte_align_empty(N, bytes, dtype=dtype)

    def zeros(N, dtype=float, bytes=16):
        return pyfftw.n_byte_align(nzeros(N, dtype=dtype), bytes)
    
    ## Monkey patches for fft
    #ifft = pyfftw.interfaces.numpy_fft.ifft
    #fft = pyfftw.interfaces.numpy_fft.fft
    #fft2 = pyfftw.interfaces.numpy_fft.fft2
    #ifft2 = pyfftw.interfaces.numpy_fft.ifft2
    #irfft = pyfftw.interfaces.numpy_fft.irfft
    #rfft = pyfftw.interfaces.numpy_fft.rfft
    #rfft2 = pyfftw.interfaces.numpy_fft.rfft2
    #irfft2 = pyfftw.interfaces.numpy_fft.irfft2
    #ifftn = pyfftw.interfaces.numpy_fft.ifftn
    #fftn = pyfftw.interfaces.numpy_fft.fftn
    #irfftn = pyfftw.interfaces.numpy_fft.irfftn
    #rfftn = pyfftw.interfaces.numpy_fft.rfftn
    
    ifft_object = {}
    irfft_object = {}
    irfft2_object = {}
    rfft2_object = {}
    fft_object = {}
    rfft_object = {}
    def ifft(a, axis=None):
        global ifft_object
        if not a.shape in ifft_object:
            b = a.copy()
            ifft_object[a.shape] = pyfftw.builders.ifft(b, axis=axis)    
            
        in_array = ifft_object[a.shape].get_input_array()
        in_array[:] = a
        return ifft_object[a.shape]()

    def irfft(a, axis=None):
        global irfft_object
        if not a.shape in irfft_object:
            b = a.copy()
            irfft_object[a.shape] = pyfftw.builders.irfft(b, axis=axis)
            
        in_array = irfft_object[a.shape].get_input_array()
        in_array[:] = a
        return irfft_object[a.shape]()

    def irfft2(a, axes=None):
        global irfft2_object
        if not a.shape in irfft2_object:
            b = a.copy()
            irfft2_object[a.shape] = pyfftw.builders.irfft2(b, axes=axes)
            
        in_array = irfft2_object[a.shape].get_input_array()
        in_array[:] = a
        return irfft2_object[a.shape]()

    def fft(a, axis=None):
        global fft_object
        if not a.shape in fft_object:
            b = a.copy()
            fft_object[a.shape] = pyfftw.builders.fft(b, axis=axis)
        
        in_array = fft_object[a.shape].get_input_array()
        in_array[:] = a
        return fft_object[a.shape]()

    def rfft(a, axis=None):
        global rfft_object
        if not a.shape in rfft_object:
            b = a.copy()
            rfft_object[a.shape] = pyfftw.builders.rfft(b, axis=axis)
            
        in_array = rfft_object[a.shape].get_input_array()
        in_array[:] = a        
        return rfft_object[a.shape]()

    def rfft2(a, axes=None):
        global rfft2_object
        if not a.shape in rfft2_object:
            b = a.copy()
            rfft2_object[a.shape] = pyfftw.builders.rfft2(b, axes=axes)
            
        in_array = rfft2_object[a.shape].get_input_array()
        in_array[:] = a        
        return rfft2_object[a.shape]()

except:    
    print Warning("Install pyfftw, it is much faster than numpy fft")
