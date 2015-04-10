import pstats
import pprint

__all__ = ['create_profile']

def create_profile(profiler, comm, MPI, rank, **params):
    profiler.disable()
    ps = pstats.Stats(profiler).sort_stats('cumulative')
    ps.print_stats(500)
    
    results = {}
    for item in ['ifftn_mpi', 
                 'fftn_mpi',
                 'irfftn',
                 'irfft2',
                 'ifft',
                 'rfftn',
                 'rfft2',
                 'fft',
                 'Alltoall',
                 'Sendrecv_replace',
                 'add_pressure_diffusion',
                 'cross1',
                 'cross2',
                 'dealias_rhs',
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
