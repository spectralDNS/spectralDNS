import pstats
import pprint

__all__ = ['create_profile']

def create_profile(profiler, comm, MPI, rank, **kw):
    profiler.disable()
    ps = pstats.Stats(profiler).sort_stats('cumulative')
    #ps.print_stats(1000)
    
    results = {}
    for item in ['ifftn', 
                 'ifft',
                 'irfftn',
                 'irfft2',
                 'irfft',
                 'rfftn',
                 'rfft2',
                 'rfft',
                 'fftn',
                 'fft',
                 'dct',
                 'Alltoall',
                 'Alltoallw',
                 'Sendrecv_replace',
                 'rollaxis',
                 'add_pressure_diffusion',
                 'cross1',
                 'cross2',
                 'Curl',
                 'Cross',
                 'project',
                 'Scatter',
                 'ComputeRHS']:
        for key, val in ps.stats.iteritems():
            if item is key[2] or "method '%s'"%item in key[2] or ".%s"%item in key[2]:
                results[item] = (comm.reduce(val[2], op=MPI.MIN, root=0),
                                 comm.reduce(val[2], op=MPI.MAX, root=0),
                                 comm.reduce(val[3], op=MPI.MIN, root=0),
                                 comm.reduce(val[3], op=MPI.MAX, root=0))
                del ps.stats[key]
                break
        
    if rank == 0:
        print "Printing profiling for total min/max cumulative min/max:"
        pprint.pprint(results)

    return results
