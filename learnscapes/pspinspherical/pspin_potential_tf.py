from __future__ import division
import numpy as np
import abc
import tensorflow as tf
from pele.potentials import BasePotential
from pele.optimize._quench import lbfgs_cpp
from learnscapes.utils import *
from learnscapes.pspinspherical import GradientDescent

def MeanFieldPSpinSphericalTF(interactions, nspins, p, dtype='float32'):
    if p == 3:
        return MeanField3SpinSphericalTF(interactions, nspins, dtype=dtype)
    elif p == 4:
        return MeanField4SpinSphericalTF(interactions, nspins, dtype=dtype)
    elif p == 5:
        return MeanField5SpinSphericalTF(interactions, nspins, dtype=dtype)
    else:
        raise Exception("BaseMeanFieldPSpinSphericalTF: p={} not implemented".format(p))

class BaseMeanFieldPSpinSphericalTF(BasePotential):
    """
    the potential has been hardcoded for p=3
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, interactions, nspins, p, dtype='float32'):
        self.dtype = dtype
        self.p = p
        self.nspins = nspins
        prf = np.power(self.nspins, (self.p-1.)/2.) if self.p > 2 else 1.
        prf *= nspins #normalize wrt to system size to get comparable values, see http://arxiv.org/pdf/1412.6615v4.pdf
        self.sqrtN = np.sqrt(self.nspins)
        # the following scheme needs to be followed to avoid repeated addition of ops to
        # the graph
        self.g = tf.Graph()
        with self.g.as_default():
            self.prf = tf.constant(prf, dtype=self.dtype, name='pot_prf')
            self.interactions = tf.constant(interactions, dtype=self.dtype, name='pot_interactions')
            self.x = tf.Variable(tf.zeros([self.nspins], dtype=self.dtype), name='x')
            #declaring loss like this makes sure that the full graph is initialised
            self.gloss= self.loss
            self.gcompute_gradient = self.compute_gradient
            self.session = tf.Session()
            init = tf.initialize_all_variables()
            self.session.run(init)
            self.g.finalize() # this guarantees that no new ops are added to the graph

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

    @property
    def lossTensorPartial(self):
        """this gives a 2x2 matrix generator"""
        return tf.mul(tf.reshape(self.x, [self.nspins]), tf.reshape(self.x, [self.nspins,1]))

    @property
    @abc.abstractmethod
    def lossTensor(self):
        """this gives the full loss tensor generator"""

    @property
    def loss(self):
        """this mutliplies times the interaction and reduces sum reduces the tensor"""
        return -tf.div(tf.reduce_sum(tf.mul(self.interactions, self.lossTensor)), self.prf)

    @property
    def compute_gradient(self):
        grad = tf.gradients(self.loss, self.x)[0]
        grad = tfOrthog(grad, tf.mul(self.x, tfRnorm(self.x)))
        return grad

    def _normalizeSpins(self, coords):
        coords /= (np.linalg.norm(coords)/self.sqrtN)

    def getEnergy(self, coords):
        self._normalizeSpins(coords)
        with self.g.as_default():
            with self.session.as_default():
                e = self.session.run(self.gloss, feed_dict={self.x : coords})
        return e

    def getEnergyGradient(self, coords):
        self._normalizeSpins(coords)
        with self.g.as_default():
            with self.session.as_default():
                e, grad = self.session.run([self.gloss, self.gcompute_gradient],
                                               feed_dict={self.x : coords})
        return e, grad

class MeanField3SpinSphericalTF(BaseMeanFieldPSpinSphericalTF):
    def __init__(self, interactions, nspins, dtype='float32'):
        super(MeanField3SpinSphericalTF, self).__init__(interactions, nspins, p=3, dtype=dtype)

    @property
    def lossTensor(self):
        """this gives the full loss tensor generator"""
        return tf.mul(self.lossTensorPartial, tf.reshape(self.x, [self.nspins,1,1]))

class MeanField4SpinSphericalTF(BaseMeanFieldPSpinSphericalTF):
    def __init__(self, interactions, nspins, dtype='float32'):
        super(MeanField4SpinSphericalTF, self).__init__(interactions, nspins, p=4, dtype=dtype)

    @property
    def lossTensor(self):
        """this gives the full loss tensor generator"""
        return tf.mul(tf.mul(self.lossTensorPartial, tf.reshape(self.x, [self.nspins,1,1])),
                      tf.reshape(self.x, [self.nspins,1,1,1]))

class MeanField5SpinSphericalTF(BaseMeanFieldPSpinSphericalTF):
    def __init__(self, interactions, nspins, dtype='float32'):
        super(MeanField5SpinSphericalTF, self).__init__(interactions, nspins, p=5, dtype=dtype)

    @property
    def lossTensor(self):
        """this gives the full loss tensor generator"""
        return tf.mul(tf.mul(tf.mul(self.lossTensorPartial, tf.reshape(self.x, [self.nspins,1,1])),
                             tf.reshape(self.x, [self.nspins,1,1,1])),
                      tf.reshape(self.x, [self.nspins,1,1,1,1]))

if __name__ == "__main__":
    import timeit
    np.random.seed(42)

    def minimize(pot, coords):
        print "start energy", pot.getEnergy(coords)
        results = lbfgs_cpp(coords, pot, M=4, nsteps=1e5, tol=1e-5, iprint=-1, verbosity=0, maxstep=n)
        print "quenched energy", results.energy
        print "E: {}, nsteps: {}".format(results.energy, results.nfev)
        print results
        if results.success:
            return [results.coords, results.energy, results.nfev]

    dtype = 'float32'
    n=50
    p=3
    norm = tf.random_normal([n for _ in xrange(p)], mean=0, stddev=1.0, dtype=dtype)
    interactions = norm.eval(session=tf.Session())

    coords = np.random.normal(size=n)
    coords /= np.linalg.norm(coords) / np.sqrt(coords.size)
    potTF = MeanFieldPSpinSphericalTF(interactions, n, p, dtype=dtype)

    e, grad = potTF.getEnergyGradient(coords)
    print '{0:.15f}'.format(e), grad

    # gd = GradientDescent(potTF, learning_rate=1, iprint=1)
    # gd.run(coords)
    # print gd.get_results()

    # print timeit.timeit('e, grad = potPL.getEnergyGradient(coords)', "from __main__ import potPL, coords", number=10)
    # import time
    # for _ in xrange(2000):
    #     start = time.time()
    #     e, grad = potTF.getEnergyGradient(coords)
    #     print time.time() - start
    # print timeit.timeit('e, grad = potTF.getEnergyGradient(coords)', "from __main__ import potTF, coords", number=1000)

    # minimize(potPL, coords)
    # print minimize(potTF, coords)
    #
    # print timeit.timeit('minimize(potPL, coords)', "from __main__ import potPL, coords, minimize", number=1)
    print timeit.timeit('minimize(potTF, coords)', "from __main__ import potTF, coords, minimize", number=10)
