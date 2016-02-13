from __future__ import division
import numpy as np
from learnscapes.NestedSampling import NSPotential
from nested_sampling import NestedSampling as ns
from nested_sampling import MonteCarloWalker, run_nested_sampling

def random_displace(x, stepsize):
    x += np.random.uniform(low=-stepsize, high=stepsize, size=x.shape)
    return x

class NestedSampling(object):
    def __init__(self, lsPotential, takeStep=random_displace, acceptTest=None,
                 nreplicas=10, mciter=10, nproc=1, verbose=False):
        self.mciter, self.nreplicas = mciter, nreplicas
        self.nproc, self.verbose = nproc, verbose
        pot = NSPotential(lsPotential)
        mcrunner = MonteCarloWalker(pot, takestep=takeStep,
                                    accept_test=acceptTest, mciter=self.mciter)
        self.ns = ns(pot, self.nreplicas, mcrunner,
                     nproc=self.nproc, verbose=self.verbose)

    def run(self, label="ns_out", etol=1e-5, maxiter=None, iprint_replicas=1000):
        return run_nested_sampling(self.ns, label=label, etol=etol,
                                   maxiter=maxiter, iprint_replicas=iprint_replicas)