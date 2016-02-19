from __future__ import division
import numpy as np
from learnscapes.pspinspherical import MeanFieldPSpinSphericalTF


class NSPotentialNN(object):
    """
    wrapper to run learnscape neural network potential in nested_sampling

    lsPotential: learnscapes potential
    """
    def __init__(self, lsPotential, scale=1):
        self.pot = lsPotential
        self.ndim = self.pot.ndim
        self.scale = scale

    def get_energy(self, x):
        return self.pot.getEnergy(x)

    def get_random_configuration(self):
        """ return a random vector sampled uniformly from within a hypersphere of dimensions self.ndim"""
        x = np.random.randn(self.ndim) * self.scale
        return x


class NSPotentialPSS(object):
    """
    wrapper to run learnscape p-spin spherical potential in nested_sampling

    lsPotential: learnscapes potential
    """
    def __init__(self, interactions, nspins, p, dtype='float64', device='gpu'):
        self.nspins, self.p = nspins, p
        self.dtype, self.device = dtype, device
        self.sqrtN = np.sqrt(nspins)
        self.pot = MeanFieldPSpinSphericalTF(interactions, nspins, p, dtype=dtype, device=device)

    def _normalize_spins(self, x):
        x /= (np.linalg.norm(x)/self.sqrtN)

    def get_energy(self, x):
        self._normalize_spins(x)
        return self.pot.getEnergy(x)

    def get_random_configuration(self):
        """ return a random vector sampled uniformly from within a hypersphere of dimensions self.ndim"""
        x = np.random.randn(self.nspins)
        self._normalize_spins(x)
        return x