from __future__ import division
import numpy as np
import tensorflow as tf
from pele.potentials import BasePotential
from pele.optimize._quench import lbfgs_cpp
from learnscapes.utils import tfRnorm, tfOrthog, select_device_simple
from learnscapes.pspinspherical import GradientDescent

#this is the underlying graph


def MeanFieldPSpinGraph(interactions, nspins, p, dtype='float32'):
    if p == 3:
        return MeanField3SpinGraph(interactions, nspins, dtype=dtype)
    elif p == 4:
        return MeanField4SpinGraph(interactions, nspins, dtype=dtype)
    elif p == 5:
        return MeanField5SpinGraph(interactions, nspins, dtype=dtype)
    else:
        raise Exception("MeanFieldPSpinGraph: p={} not implemented".format(p))


class BasePSpinGraph(object):
    """
    the only missing function is normalize gradient that is not implemented as part of the graph
    this should not have its own graph or its own section, it defines only the graph
    """

    def __init__(self, interactions, nspins, p, dtype='float32'):
        self.dtype = dtype
        self.p = p
        self.nspins = nspins
        self.pyinteractions = interactions
        # normalize wrt to system size to get comparable values,
        # see http://arxiv.org/pdf/1412.6615v4.pdf
        prf = np.power(self.nspins, (self.p-1.)/2.) if self.p > 2 else 1.
        self.pyprf = prf * nspins

    def __call__(self, graph=tf.Graph()):
        """
        the following scheme needs to be followed to avoid repeated addition of ops to the graph
        :param graph:
        :return:
        """
        self.g = graph
        with self.g.name_scope('embedding'):
            self.prf = tf.constant(self.pyprf, dtype=self.dtype, name='pot_prf')
            self.interactions = tf.constant(self.pyinteractions, dtype=self.dtype, name='pot_interactions')
            self.x = tf.Variable(tf.zeros([self.nspins], dtype=self.dtype), name='x')
        # declaring loss like this makes sure that the full graph is initialised
        with self.g.name_scope('loss'):
            self.gloss= self.loss
        with self.g.name_scope('gradient'):
            self.gcompute_gradient = self.compute_gradient
        return self

    @property
    def lossTensorPartial(self):
        """this gives a 2x2 matrix generator"""
        return tf.mul(tf.reshape(self.x, [self.nspins]), tf.reshape(self.x, [self.nspins,1]))

    @property
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


class MeanField3SpinGraph(BasePSpinGraph):
    def __init__(self, interactions, nspins, dtype='float32'):
        super(MeanField3SpinGraph, self).__init__(interactions, nspins, 3, dtype=dtype)

    @property
    def lossTensor(self):
        """this gives the full loss tensor generator"""
        return tf.mul(self.lossTensorPartial, tf.reshape(self.x, [self.nspins,1,1]))


class MeanField4SpinGraph(BasePSpinGraph):
    def __init__(self, interactions, nspins, dtype='float32'):
        super(MeanField4SpinGraph, self).__init__(interactions, nspins, 4, dtype=dtype)

    @property
    def lossTensor(self):
        """this gives the full loss tensor generator"""
        return tf.mul(tf.mul(self.lossTensorPartial, tf.reshape(self.x, [self.nspins,1,1])),
                      tf.reshape(self.x, [self.nspins,1,1,1]))


class MeanField5SpinGraph(BasePSpinGraph):
    def __init__(self, interactions, nspins, dtype='float32'):
        super(MeanField5SpinGraph, self).__init__(interactions, nspins, 5, dtype=dtype)

    @property
    def lossTensor(self):
        """this gives the full loss tensor generator"""
        return tf.mul(tf.mul(tf.mul(self.lossTensorPartial, tf.reshape(self.x, [self.nspins,1,1])),
                             tf.reshape(self.x, [self.nspins,1,1,1])),
                      tf.reshape(self.x, [self.nspins,1,1,1,1]))


class MeanFieldPSpinSphericalTF(BasePotential):
    """
    the potential has been hardcoded for p=3
    """
    def __init__(self, interactions, nspins, p, dtype='float32', device='gpu'):
        self.sqrtN = np.sqrt(nspins)
        self.device = select_device_simple(dev=device)
        # the following scheme needs to be followed to avoid repeated addition of ops to
        # the graph
        self.g = tf.Graph()
        with self.g.as_default(), self.g.device(self.device):
            self.session = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False))
            with self.session.as_default():
                self.model = MeanFieldPSpinGraph(interactions, nspins, p, dtype=dtype)(graph=self.g)
                init = tf.initialize_all_variables()
                self.session.run(init)
                self.g.finalize()   # this guarantees that no new ops are added to the graph

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

    def _normalizeSpins(self, coords):
        coords /= (np.linalg.norm(coords)/self.sqrtN)

    def getEnergy(self, coords):
        self._normalizeSpins(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                e = self.session.run(self.model.gloss, feed_dict={self.model.x: coords})
        return e

    def getEnergyGradient(self, coords):
        self._normalizeSpins(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                e, grad = self.session.run([self.model.gloss, self.model.gcompute_gradient],
                                           feed_dict={self.model.x: coords})
        return e, grad

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
    n=100
    p=3
    norm = tf.random_normal([n for _ in xrange(p)], mean=0, stddev=1.0, dtype=dtype)
    interactions = norm.eval(session=tf.Session())

    coords = np.random.normal(size=n)
    coords /= np.linalg.norm(coords) / np.sqrt(coords.size)

    potTF = MeanFieldPSpinSphericalTF(interactions, n, p, dtype=dtype, device='gpu')

    e, grad = potTF.getEnergyGradient(coords)
    print 'e:{0:.15f}, norm(g):{0:.15f}'.format(e), np.linalg.norm(grad)


    # model = MeanFieldPSpinGraph(interactions, n, p, dtype=dtype)
    # gd = GradientDescent(model, learning_rate=1, iprint=1, device='cpu')
    # gd.run(coords)
    # print gd.get_results()

    # import time
    # for _ in xrange(2000):
    #     start = time.time()
    #     e, grad = potTF.getEnergyGradient(coords)
    #     print time.time() - start
    # print timeit.timeit('e, grad = potTF.getEnergyGradient(coords)', "from __main__ import potTF, coords", number=10)

    # print minimize(potTF, coords)
    #
    # print timeit.timeit('minimize(potTF, coords)', "from __main__ import potTF, coords, minimize", number=10)
