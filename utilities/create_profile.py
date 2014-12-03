import pstats
import pprint

def create_profile(profiler, comm, MPI, rank, **params):
    profiler.disable()
    ps = pstats.Stats(profiler).sort_stats('cumulative')
    #ps.print_stats(make_profile)
    
    results = {}
    for item in ['ifftn_mpi', 
                 'fftn_mpi',
                 '_Xfftn',
                 'Alltoall',
                 'Sendrecv_replace',
                 'Curl',
                 'project',
                 'Scatter',
                 'ComputeRHS']:
        for key, val in ps.stats.iteritems():
            if item in key[2]:
                results[item] = (comm.reduce(val[2], op=MPI.MIN, root=0),
                                 comm.reduce(val[2], op=MPI.MAX, root=0),
                                 comm.reduce(val[3], op=MPI.MIN, root=0),
                                 comm.reduce(val[3], op=MPI.MAX, root=0))
                break
        del ps.stats[key]
        
    if rank == 0:
        print "Printing profiling for total min/max cumulative min/max:"
        pprint.pprint(results)

    return results

