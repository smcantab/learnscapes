from __future__ import division
import numpy as np
import nested_sampling
from nested_sampling import MonteCarloWalker, run_nested_sampling, Replica


def random_displace(x, stepsize):
    x += np.random.uniform(low=-stepsize, high=stepsize, size=x.shape)
    return x


class NestedSampling(object):
    def __init__(self, nsPotential, scale=1, takeStep=random_displace,
                 acceptTest=None, nreplicas=10, mciter=10, nproc=1,
                 verbose=False):
        self.mciter, self.nreplicas = mciter, nreplicas
        self.nproc, self.verbose = nproc, verbose
        self.pot = nsPotential
        mcrunner = MonteCarloWalker(self.pot, takestep=takeStep,
                                    accept_test=acceptTest, mciter=self.mciter)
        replicas = self.initialise_replicas()
        self.ns = nested_sampling.NestedSampling(replicas, mcrunner,
                                                 nproc=self.nproc, verbose=self.verbose)

    def initialise_replicas(self):
        # create the replicas
        replicas = []
        for i in xrange(self.nreplicas):
            x = self.pot.get_random_configuration()
            e = self.pot.get_energy(x)
            replicas.append(Replica(x, e))
        return replicas

    def run(self, label="ns_out", etol=1e-5, maxiter=None, iprint_replicas=1000):
        return run_nested_sampling(self.ns, label=label, etol=etol,
                                   maxiter=maxiter, iprint_replicas=iprint_replicas)