from shenfun import HDF5File as H5File

__all__ = ['HDF5File']

#pylint: disable=dangerous-default-value,unused-argument


class HDF5File(object):
    """Class for storing and retrieving spectralDNS data

    The class stores two types of data

        - checkpoint
        - results

    Checkpoint data are used to store intermediate simulation results, and can
    be used to restart a simulation at a later stage, with no loss of accuracy.

    Results data are used for visualization.

    Data is provided as dictionaries. The checkpoint dictionary is represented
    as::

        checkpoint = {'space': T,
                      'data': {
                          '0': {'U': [U_hat]},
                          '1': {'U': [U0_hat]},
                          ...
                          }
                      }

    where T is the function space of the data to be stored, and 'data' contains
    solutions to be stored at possibly several different timesteps. The current
    timestep is 0, previous is 1 and so on if more is needed by the integrator.
    Note that checkpoint is storing results from spectral space, i.e., the
    output of a forward transform of the space.

    The results dictionary is like::

        results = {'space': T,
                   'data': {
                       'U': [U, (U, [slice(None), slice(None), 0])],
                       'V': [V, (V, [slice(None), 0, slice(None)])],
                       }
                   }

    The results will be stored as scalars, even if U and V are vectors. Results
    are store for physical space, i.e., the input to a forward transform of the
    space.

    """

    def __init__(self, filename, checkpoint={}, results={}):
        self.cfile = None
        self.wfile = None
        self.filename = filename
        self.checkpoint = checkpoint
        self.results = results

    def update(self, params, **kw):
        if self.cfile is None:
            self.cfile = H5File(self.filename+'_c.h5', self.checkpoint['space'], mode='a')
        if self.wfile is None:
            self.wfile = H5File(self.filename+'_w.h5', self.results['space'], mode='a')

        if params.tstep % params.write_result == 0:
            self.update_components(**kw)
            self.wfile.write(params.tstep, self.results['data'], as_scalar=True, forward_output=False)

        if params.tstep % params.checkpoint == 0:
            for key, val in self.checkpoint['data'].items():
                self.cfile.write(int(key), val, forward_output=True)

    def update_components(self, **kw):
        pass

    def open(self):
        self.cfile.open()
        self.wfile.open()

    def close(self):
        if self.cfile.f:
            self.cfile.close()
        if self.wfile.f:
            self.wfile.close()
