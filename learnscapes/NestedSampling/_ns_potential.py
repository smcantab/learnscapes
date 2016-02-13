from __future__ import division
import numpy as np

class NSPotential(object):
    """
    wrapper to run learnscape potential in nested_sampling

    lsPotential: learnscapes potential
    """
    def __init__(self, lsPotential, scale):
        self.pot = lsPotential
        self.ndim = self.pot.ndim
        self.scale = scale

    def get_energy(self, x):
        return self.pot.getEnergy(x)

    def get_random_configuration(self):
        """ return a random vector sampled uniformly from within a hypersphere of dimensions self.ndim"""
        x = np.random.randn(self.ndim) * self.scale
        return x