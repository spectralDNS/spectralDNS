import pstats

def create_profile(profile, comm, MPI, **params):
    profile.disable()
    ps = pstats.Stats(profile).sort_stats('cumulative')
    #ps.print_stats(make_profile)
    
    results = {}
    for item in ['fftn_mpi', 
                 'ifftn_mpi', 
                 '_Xfftn',
                 'Alltoall',
                 'Sendrecv_replace',
                 'Curl',
                 'project',
                 'ComputeRHS']:
        for key, val in ps.stats.iteritems():
            if item in key[2]:
                results[item] = (comm.reduce(val[2], op=MPI.MIN, root=0),
                                 comm.reduce(val[2], op=MPI.MAX, root=0),
                                 comm.reduce(val[3], op=MPI.MIN, root=0),
                                 comm.reduce(val[3], op=MPI.MAX, root=0))
    return results

