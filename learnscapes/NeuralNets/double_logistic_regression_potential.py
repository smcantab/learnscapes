from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pele.potentials import BasePotential
from pele.optimize._quench import lbfgs_cpp
from learnscapes.NeuralNets import BaseMLGraph
from learnscapes.utils import select_device_simple


class DoubleLogisticRegressionGraph(BaseMLGraph):
    """
    this is a basic mlp, think 2 stacked logistic regressions
    :param hnodes: number of hidden nodes
    """
    def __init__(self, x_train_data, y_train_data, hnodes, reg=0, dtype='float64'):
        super(DoubleLogisticRegressionGraph, self).__init__(x_train_data, y_train_data, reg=reg,dtype=dtype)
        self.hnodes = hnodes

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
            self.x_test = tf.placeholder(self.dtype, shape=(None, self.shape[0]), name='x_test_data')
            self.y_test = tf.placeholder(self.dtype, shape=(None, self.shape[1]), name='y_test_data')
            self.w_h = tf.Variable(tf.zeros((self.shape[0], self.hnodes), dtype=self.dtype), name='hidden_layer_weights')
            self.w_o = tf.Variable(tf.zeros((self.hnodes, self.shape[1]), dtype=self.dtype), name='output_layer_weights')
            self.b_h = tf.Variable(tf.zeros([self.hnodes], dtype=self.dtype), name='hidden_layer_bias')
            self.b_o = tf.Variable(tf.zeros([self.shape[1]], dtype=self.dtype), name='output_layer_bias')
        # declaring loss like this makes sure that the full graph is initialised
        with self.g.name_scope('loss'):
            self.gloss= self.regularised_loss
        with self.g.name_scope('gradient'):
            self.gcompute_gradient = self.compute_gradient
        with self.g.name_scope('predict'):
            self.gpredict = self.predict
        return self

    @property
    def model(self):
        h = tf.nn.sigmoid(tf.matmul(self.x, self.w_h) + self.b_h)
        return tf.matmul(h, self.w_o) + self.b_o

    @property
    def regularization(self):
        return self.reg * (tf.nn.l2_loss(self.w_h) + tf.nn.l2_loss(self.w_o))

    @property
    def compute_gradient(self):
        return tf.gradients(self.regularised_loss, [self.w_h, self.w_o, self.b_h, self.b_o])

    # @property
    # def compute_hessian(self):
    #     n = self.shape[0]*self.hnodes+self.hnodes*self.shape[1]+self.hnodes+self.shape[1]
    #     n1 = self.shape[0]*self.hnodes
    #     for i in xrange(n1):
    #         for j in xrange(i, n1):
    #             self.hess[i, ]
    #     n2 = n1 + self.hnodes*self.shape[1]
    #     for i in xrange(n1,n2):
    #         for j in xrange(i, n2):
    #             # run something
    #     n3 = n2 + self.hnodes
    #     for i in xrange(n2,n3):
    #         for j in xrange(i, n3):
    #             # run something
    #     n4 = n3 + self.shape[1]
    #     for i in xrange(n3,n4):
    #         for j in xrange(i, n4):
    #             # run something

    @property
    def predict(self):
        """this tests the models, at predict time, evaluate the argmax of the logistic regression
        """
        h = tf.nn.sigmoid(tf.matmul(self.x_test, self.w_h) + self.b_h)
        model_test = tf.matmul(h, self.w_o) + self.b_o
        return tf.argmax(model_test, 1)


class DoubleLogisticRegressionPotential(BasePotential):
    """
    potential that can be used with pele toolbox
    """
    def __init__(self, x_train_data, y_train_data, hnodes, reg=0, dtype='float64', device='cpu'):
        self.device = select_device_simple(dev=device)
        # the following scheme needs to be followed to avoid repeated addition of ops to
        # the graph
        self.g = tf.Graph()
        with self.g.as_default(), self.g.device(self.device):
            self.session = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False))
            with self.session.as_default():
                self.tf_graph = DoubleLogisticRegressionGraph(x_train_data, y_train_data,
                                                              hnodes, reg=reg, dtype=dtype)(graph=self.g)
                init = tf.initialize_all_variables()
                self.session.run(init)
                self.g.finalize()   # this guarantees that no new ops are added to the graph

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

    def pele_to_tf(self, coords):
        featdim, outdim = self.tf_graph.shape
        hnodes = self.tf_graph.hnodes
        w_h = coords[:featdim*hnodes].reshape((featdim, hnodes))
        w_o = coords[featdim*hnodes:-(hnodes+outdim)].reshape((hnodes, outdim))
        b_h = coords[-(hnodes+outdim):-outdim]
        b_o = coords[-outdim:]
        return w_h, w_o, b_h, b_o

    def tf_to_pele(self, w_h, w_o, b_h, b_o):
        w = np.append(w_h.flatten(), w_o.flatten())
        b = np.append(b_h, b_o)
        return np.append(w, b)

    def getEnergy(self, coords):
        w_h, w_o, b_h, b_o = self.pele_to_tf(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                e = self.session.run(self.tf_graph.gloss, feed_dict={self.tf_graph.w_h: w_h,
                                                                     self.tf_graph.w_o: w_o,
                                                                     self.tf_graph.b_h: b_h,
                                                                     self.tf_graph.b_o: b_o})
        return e

    def getEnergyGradient(self, coords):
        w_h, w_o, b_h, b_o = self.pele_to_tf(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                operations = [self.tf_graph.gloss] + self.tf_graph.gcompute_gradient #extend operations
                e, grad_w_h, grad_w_o, grad_b_h, grad_b_o = self.session.run(operations,
                                                                             feed_dict={self.tf_graph.w_h: w_h,
                                                                                        self.tf_graph.w_o: w_o,
                                                                                        self.tf_graph.b_h: b_h,
                                                                                        self.tf_graph.b_o: b_o})
        return e, self.tf_to_pele(grad_w_h, grad_w_o,
                                  grad_b_h, grad_b_o)

    # def getEnergyGradientHessian(self, coords):
    #     e, g = self.getEnergyGradient(coords)
    #     h = self.NumericalHessian(coords, eps=1e-6)
    #     return e, g, h


    def test_model(self, coords, x_test, y_test):
        w_h, w_o, b_h, b_o = self.pele_to_tf(coords)
        prediction = self.session.run([self.tf_graph.gpredict],
                                      feed_dict={self.tf_graph.w_h: w_h,
                                                 self.tf_graph.w_o: w_o,
                                                 self.tf_graph.b_h: b_h,
                                                 self.tf_graph.b_o: b_o,
                                                 self.tf_graph.x_test: x_test,
                                                 self.tf_graph.y_test: y_test})
        return np.mean(np.argmax(y_test, axis=1) == prediction)


def main():
    np.random.seed(42)

    def minimize(pot, coords):
        print "shape", coords.shape
        print "start energy", pot.getEnergy(coords)
        results = lbfgs_cpp(coords, pot, M=4, nsteps=1e5, tol=1e-5, iprint=1, verbosity=1, maxstep=10)
        print "quenched energy", results.energy
        print "E: {}, nsteps: {}".format(results.energy, results.nfev)
        if results.success:
            # return [results.coords, results.energy, results.nfev]
            return results

    def init_weights(shape):
        return np.random.normal(0, scale=0.01, size=shape).flatten()

    dtype = 'float64'
    device = 'gpu'

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    bs = 1000
    trX, trY, teX, teY = mnist.train.images[:bs], mnist.train.labels[:bs], mnist.test.images, mnist.test.labels

    # like in linear regression, we need a shared variable weight matrix for logistic regression
    hnodes = 10 #625
    w_h = init_weights([784, hnodes])
    w_o = init_weights([hnodes, 10])
    w = np.append(w_h, w_o)
    b_h = np.ones(hnodes)*0.1
    b_o = np.ones(10)*0.1
    b = np.append(b_h, b_o)
    weights = np.append(w, b)
    potTF = DoubleLogisticRegressionPotential(trX, trY, hnodes, reg=0.1, dtype=dtype, device=device)

    e, grad = potTF.getEnergyGradient(weights)
    print 'e:{0:.15f}, norm(g):{0:.15f}'.format(e, np.linalg.norm(grad))

    results = minimize(potTF, weights)
    # print potTF.test_model(np.array(results.coords), teX, teY)

    from pele.transition_states import findLowestEigenVector, analyticalLowestEigenvalue
    res = findLowestEigenVector(np.array(results.coords), potTF, orthogZeroEigs=None, iprint=1, tol=1e-6)
    print res.H0
    print analyticalLowestEigenvalue(np.array(results.coords), potTF)

if __name__ == "__main__":
    main()