from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pele.potentials import BasePotential
from pele.optimize._quench import lbfgs_cpp
from learnscapes.utils import tfRnorm, tfOrthog, select_device_simple

#this is the underlying graph

class BaseRegressionGraph(object):
    """
    this should not have its own graph or its own section, it defines only the graph
    """

    def __init__(self, x_train_data, y_train_data, dtype='float32'):
        self.dtype = dtype
        self.py_x = np.array(x_train_data)
        self.py_y = np.array(y_train_data)

    def __call__(self, graph=tf.Graph()):
        """
        the following scheme needs to be followed to avoid repeated addition of ops to the graph
        :param graph:
        :return:
        """
        self.g = graph
        with self.g.name_scope('embedding'):
            self.x = tf.constant(self.py_x, dtype=self.dtype, name='x_training_data')
            self.y = tf.constant(self.py_y, dtype=self.dtype, name='y_training_data')
            self.w = tf.Variable(tf.zeros(self.shape, dtype=self.dtype), name='weights')
        # declaring loss like this makes sure that the full graph is initialised
        with self.g.name_scope('loss'):
            self.gloss= self.loss
        with self.g.name_scope('gradient'):
            self.gcompute_gradient = self.compute_gradient
        return self

    @property
    def model(self):
        """
        this is the model, it could be linear or quadratic etc
        """

    @property
    def loss(self):
        """this mutliplies times the interaction and reduces sum reduces the tensor"""

    @property
    def compute_gradient(self):
        return tf.gradients(self.loss, [self.w])[0]

class LinearRegressionGraph(BaseRegressionGraph):
    """
    linear regression model of the form y = w*x, so this model is very simple
    """
    def __init__(self, x_train_data, y_train_data, dtype='float32'):
        super(LinearRegressionGraph, self).__init__(x_train_data, y_train_data, dtype=dtype)
        assert self.py_x.shape == self.py_y.shape
        self.shape = self.py_x.shape

    @property
    def model(self):
        return tf.mul(self.x, self.w)

    @property
    def loss(self):
        """this mutliplies times the interaction and reduces sum reduces the tensor"""
        return tf.reduce_sum(tf.pow(self.y-self.model, 2)) # use sqr error for cost function

class LogisticRegressionGraph(BaseRegressionGraph):
    """
    linear regression model of the form y = w*x, so this model is very simple
    """
    def __init__(self, x_train_data, y_train_data, dtype='float32'):
        super(LogisticRegressionGraph, self).__init__(x_train_data, y_train_data, dtype=dtype)
        assert self.py_y.shape[0] == self.py_x.shape[0]
        self.shape = (self.py_x.shape[1], self.py_y.shape[1])
        print self.shape

    @property
    def model(self):
        return tf.matmul(self.x, self.w)

    @property
    def loss(self):
        """this mutliplies times the interaction and reduces sum reduces the tensor
            compute mean cross entropy (softmax is applied internally)
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.y))

class RegressionPotential(BasePotential):
    """
    potential that can be used with pele toolbox
    """
    def __init__(self, x_train_data, y_train_data, graph_type, dtype='float32', device='cpu'):
        self.device = select_device_simple(dev=device)
        # the following scheme needs to be followed to avoid repeated addition of ops to
        # the graph
        self.g = tf.Graph()
        with self.g.as_default(), self.g.device(self.device):
            self.session = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False))
            with self.session.as_default():
                self.tf_graph = graph_type(x_train_data, y_train_data, dtype=dtype)(graph=self.g)
                init = tf.initialize_all_variables()
                self.session.run(init)
                self.g.finalize()   # this guarantees that no new ops are added to the graph

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

    def pele_to_tf(self, coords):
        # featdim, outdim = self.tf_graph.shape
        # print "featdim, outdim", featdim, outdim
        # w = coords[:featdim*outdim].reshape((featdim, outdim))
        w = coords.reshape(self.tf_graph.shape)
        return w

    def tf_to_pele(self, w):
        return w.flatten()

    def getEnergy(self, coords):
        coords = self.pele_to_tf(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                e = self.session.run(self.tf_graph.gloss, feed_dict={self.tf_graph.w: coords})
        return e

    def getEnergyGradient(self, coords):
        coords = self.pele_to_tf(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                e, grad = self.session.run([self.tf_graph.gloss, self.tf_graph.gcompute_gradient],
                                           feed_dict={self.tf_graph.w: coords})
        return e, self.tf_to_pele(grad)

if __name__ == "__main__":
    import timeit
    np.random.seed(42)

    def minimize(pot, coords):
        print "shape", coords.shape
        print "start energy", pot.getEnergy(coords)
        results = lbfgs_cpp(coords, pot, M=4, nsteps=1e5, tol=1e-5, iprint=1, verbosity=0, maxstep=10)
        print "quenched energy", results.energy
        print "E: {}, nsteps: {}".format(results.energy, results.nfev)
        print results
        if results.success:
            return [results.coords, results.energy, results.nfev]

    def init_weights(shape):
        return np.random.normal(0, scale=0.01, size=shape).flatten()

    dtype = 'float32'

    # linear regression
    if False:
        trX = np.linspace(-1, 1, 101)
        trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise
        weights = np.zeros(trX.shape)
        potTF = RegressionPotential(trX, trY, LinearRegressionGraph, dtype=dtype, device='cpu')

    if True:
        # like in linear regression, we need a shared variable weight matrix for logistic regression
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        bs = 50000
        trX, trY, teX, teY = mnist.train.images[:bs], mnist.train.labels[:bs], mnist.test.images, mnist.test.labels
        weights = init_weights([784, 10])
        potTF = RegressionPotential(trX, trY, LogisticRegressionGraph, dtype=dtype, device='cpu')

    e, grad = potTF.getEnergyGradient(weights)
    print 'e:{0:.15f}, norm(g):{0:.15f}'.format(e, np.linalg.norm(grad))


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

    print minimize(potTF, weights)
    #
    # print timeit.timeit('minimize(potTF, coords)', "from __main__ import potTF, coords, minimize", number=10)
