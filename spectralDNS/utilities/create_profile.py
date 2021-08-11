"""
Module for inspecting profiler in spectralDNS.
"""
import pstats
import pprint
from mpi4py import MPI

__all__ = ['create_profile', 'reset_profile']

def create_profile(profiler):
    """Inspect profiler and return a dictionary of most important results

    args:
        profiler          Instance of cProfile.Profile()
    """
    profiler.disable()
    ps = pstats.Stats(profiler).sort_stats('cumulative')
    #ps.print_stats(1000)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

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
                 'ifst',
                 'ifct',
                 'fst',
                 'fct',
                 'Alltoall',
                 'Alltoallw',
                 'Sendrecv_replace',
                 'rollaxis',
                 'copy_to_padded',
                 'copy_from_padded',
                 'RK4',
                 'ForwardEuler',
                 'AB2',
                 'adaptiveRK',
                 'nonlinear',
                 'add_linear',
                 'cross1',
                 'cross2',
                 'compute_curl',
                 'Cross',
                 'project',
                 'Scatter',
                 'ComputeRHS',
                 'solve_linear',
                 'Conv']:
        for key, val in ps.stats.items():
            if item is key[2] or "method '%s'"%item in key[2] or ".%s"%item in key[2]:
                results[item] = (comm.reduce(val[2], op=MPI.MIN, root=0),
                                 comm.reduce(val[2], op=MPI.MAX, root=0),
                                 comm.reduce(val[3], op=MPI.MIN, root=0),
                                 comm.reduce(val[3], op=MPI.MAX, root=0))
                del ps.stats[key]
                break

    if rank == 0:
        print("Printing profiling for total min/max cumulative min/max:")
        print(" {0:14s}{1:11s}{2:11s}{3:11s}{4:11s}".format('Method', 'total min', 'total max', 'cum min', 'cum max'))
        pprint.pprint(["{0:12s} {1:2.4e} {2:2.4e} {3:2.4e} {4:2.4e}".format(k, *v)
                       for k, v in results.items()])

    return results

def reset_profile(prof):
    """Reset profiler

    args:
        prof          Instance of cProfile.Profile()
    """
    prof.code_map = {}
    prof.last_time = {}
    prof.enable_count = 0
    for func in prof.functions:
        prof.add_function(func)
