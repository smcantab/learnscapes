from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pele.potentials import BasePotential
from pele.optimize._quench import lbfgs_cpp
from learnscapes.NeuralNets import BaseMLGraph
from learnscapes.utils import select_device_simple, l2_loss


class Elu3NNGraph(BaseMLGraph):
    """
    this is a 3-layer modern NN that uses exponential activation functions
    :param hnodes: number of hidden nodes
    """
    def __init__(self, x_train_data, y_train_data, hnodes, hnodes2, reg=0, dtype='float32'):
        super(Elu3NNGraph, self).__init__(x_train_data, y_train_data, reg=reg,dtype=dtype)
        self.hnodes = hnodes
        self.hnodes2 = hnodes2
        self.ndim = (x_train_data.shape[1]*self.hnodes +
                     self.hnodes * self.hnodes2 +
                     self.hnodes2*y_train_data.shape[1] +
                     self.hnodes + self.hnodes2 +
                     y_train_data.shape[1])

    def __call__(self, graph=tf.Graph()):
        """
        the following scheme needs to be followed to avoid repeated addition of ops to the graph
        :param graph:
        :return:
        """
        self.g = graph
        with self.g.name_scope('input'):
            self.x_init = tf.placeholder(dtype=self.dtype, shape=self.py_x.shape, name='x_training_data_init')
            self.y_init = tf.placeholder(dtype=self.dtype, shape=self.py_y.shape, name='y_training_data_init')
            self.reg_init = tf.placeholder(dtype=self.dtype, shape=(), name='reg_init')
            self.x = tf.Variable(self.x_init, trainable=False, collections=[], name='x_training_data')
            self.y = tf.Variable(self.y_init, trainable=False, collections=[], name='y_training_data')
            self.reg = tf.Variable(self.reg_init, trainable=False, collections=[], name='reg')
        with self.g.name_scope('test_input'):
            self.x_test = tf.placeholder(self.dtype, shape=(None, self.shape[0]), name='x_test_data')
            self.y_test = tf.placeholder(self.dtype, shape=(None, self.shape[1]), name='y_test_data')
        with self.g.name_scope('embedding'):
            self.w_h = tf.Variable(tf.zeros((self.shape[0], self.hnodes), dtype=self.dtype), name='hidden_layer_weights')
            self.w_h2 = tf.Variable(tf.zeros((self.hnodes, self.hnodes2), dtype=self.dtype), name='hidden_layer_weights2')
            self.w_o = tf.Variable(tf.zeros((self.hnodes2, self.shape[1]), dtype=self.dtype), name='output_layer_weights')
            self.b_h = tf.Variable(tf.zeros([self.hnodes], dtype=self.dtype), name='hidden_layer_bias')
            self.b_h2 = tf.Variable(tf.zeros([self.hnodes2], dtype=self.dtype), name='hidden_layer_bias2')
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
        h = tf.nn.elu(tf.add(tf.matmul(self.x, self.w_h), self.b_h))
        h2 = tf.nn.elu(tf.add(tf.matmul(h, self.w_h2), self.b_h2))
        return tf.add(tf.matmul(h2, self.w_o), self.b_o)

    @property
    def regularization(self):
        return tf.mul(self.reg, tf.add_n([l2_loss(self.w_h), l2_loss(self.w_h2), l2_loss(self.w_o)]))

    @property
    def compute_gradient(self):
        return tf.gradients(self.regularised_loss, [self.w_h, self.w_h2, self.w_o, self.b_h, self.b_h2, self.b_o])

    @property
    def predict(self):
        """this tests the models, at predict time, evaluate the argmax of the logistic regression
        """
        h = tf.nn.elu(tf.add(tf.matmul(self.x_test, self.w_h), self.b_h))
        h2 = tf.nn.elu(tf.add(tf.matmul(h, self.w_h2), self.b_h2))
        model_test = tf.add(tf.matmul(h2, self.w_o), self.b_o)
        return tf.argmax(model_test, 1)


class Elu3NNPotential(BasePotential):
    """
    potential that can be used with pele toolbox
    """
    def __init__(self, x_train_data, y_train_data, hnodes, hnodes2, reg=0, dtype='float32', device='cpu'):
        self.device = select_device_simple(dev=device)
        # the following scheme needs to be followed to avoid repeated addition of ops to
        # the graph
        self.g = tf.Graph()
        with self.g.as_default(), self.g.device(self.device):
            self.session = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False))
            with self.session.as_default():
                self.tf_graph = Elu3NNGraph(x_train_data, y_train_data, hnodes,
                                            hnodes2, reg=reg, dtype=dtype)(graph=self.g)
                init = tf.initialize_all_variables()
                self.session.run(init)
                self.session.run(self.tf_graph.x.initializer, feed_dict={self.tf_graph.x_init:x_train_data})
                self.session.run(self.tf_graph.y.initializer, feed_dict={self.tf_graph.y_init:y_train_data})
                self.session.run(self.tf_graph.reg.initializer, feed_dict={self.tf_graph.reg_init:reg})
                self.g.finalize() # this guarantees that no new ops are added to the graph
        self.ndim = self.tf_graph.ndim

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

    def pele_to_tf(self, coords):
        featdim, outdim = self.tf_graph.shape
        hnodes = self.tf_graph.hnodes
        hnodes2 = self.tf_graph.hnodes2
        s = featdim*hnodes
        s2 = s + hnodes*hnodes2
        s3 = s2 + hnodes2*outdim
        s4 = s3 + hnodes
        s5 = s4 + hnodes2
        assert coords.size-s5 == outdim
        w_h = coords[:s].reshape((featdim, hnodes))
        w_h2 = coords[s:s2].reshape((hnodes, hnodes2))
        w_o = coords[s2:s3].reshape((hnodes2, outdim))
        b_h = coords[s3:s4]
        b_h2 = coords[s4:s5]
        b_o = coords[-outdim:]
        return w_h, w_h2, w_o, b_h, b_h2, b_o

    def tf_to_pele(self, w_h, w_h2, w_o, b_h, b_h2, b_o):
        w = np.concatenate((w_h.flatten(), w_h2.flatten(), w_o.flatten()))
        b = np.concatenate((b_h, b_h2, b_o))
        return np.append(w, b)

    def getEnergy(self, coords):
        w_h, w_h2, w_o, b_h, b_h2, b_o = self.pele_to_tf(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                e = self.session.run(self.tf_graph.gloss, feed_dict={self.tf_graph.w_h: w_h,
                                                                     self.tf_graph.w_h2: w_h2,
                                                                     self.tf_graph.w_o: w_o,
                                                                     self.tf_graph.b_h: b_h,
                                                                     self.tf_graph.b_h2: b_h2,
                                                                     self.tf_graph.b_o: b_o})
        return e

    def getEnergyGradient(self, coords):
        w_h, w_h2, w_o, b_h, b_h2, b_o = self.pele_to_tf(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                operations = [self.tf_graph.gloss] + self.tf_graph.gcompute_gradient #extend operations
                e, grad_w_h, grad_w_h2, grad_w_o, \
                grad_b_h, grad_b_h2, grad_b_o = self.session.run(operations,
                                                                 feed_dict={self.tf_graph.w_h: w_h,
                                                                            self.tf_graph.w_h2: w_h2,
                                                                            self.tf_graph.w_o: w_o,
                                                                            self.tf_graph.b_h: b_h,
                                                                            self.tf_graph.b_h2: b_h2,
                                                                            self.tf_graph.b_o: b_o})
        return e, self.tf_to_pele(grad_w_h, grad_w_h2, grad_w_o,
                                  grad_b_h, grad_b_h2, grad_b_o)

    def test_model(self, coords, x_test, y_test):
        w_h, w_h2, w_o, b_h, b_h2, b_o = self.pele_to_tf(coords)
        prediction = self.session.run([self.tf_graph.gpredict],
                                      feed_dict={self.tf_graph.w_h: w_h,
                                                 self.tf_graph.w_h2: w_h2,
                                                 self.tf_graph.w_o: w_o,
                                                 self.tf_graph.b_h: b_h,
                                                 self.tf_graph.b_h2: b_h2,
                                                 self.tf_graph.b_o: b_o,
                                                 self.tf_graph.x_test: x_test,
                                                 self.tf_graph.y_test: y_test})
        return np.mean(np.argmax(y_test, axis=1) == prediction)


def main():
    np.random.seed(42)

    def minimize(pot, coords):
        print "shape", coords.shape
        print "start energy", pot.getEnergy(coords)
        results = lbfgs_cpp(coords, pot, M=4, nsteps=1e5, tol=1e-9, iprint=1, verbosity=1, maxstep=10)
        print "quenched energy", results.energy
        print "E: {}, nsteps: {}".format(results.energy, results.nfev)
        if results.success:
            # return [results.coords, results.energy, results.nfev]
            return results

    def init_weights(shape):
        return np.random.normal(0, scale=1, size=shape).flatten()

    dtype = 'float32'
    device = 'cpu'

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    bs = 100
    trX, trY, teX, teY = mnist.train.images[:bs], mnist.train.labels[:bs], mnist.test.images, mnist.test.labels

    # like in linear regression, we need a shared variable weight matrix for logistic regression
    hnodes = 10
    hnodes2 = 10
    w_h = init_weights([784, hnodes])
    w_h2 = init_weights([hnodes, hnodes2])
    w_o = init_weights([hnodes2, 10])
    w = np.concatenate((w_h, w_h2, w_o))
    b_h = np.ones(hnodes)*0.1
    b_h2 = np.ones(hnodes2)*0.1
    b_o = np.ones(10)*0.1
    b = np.concatenate((b_h, b_h2, b_o))
    weights = np.append(w, b)
    potTF = Elu3NNPotential(trX, trY, hnodes, hnodes2, dtype=dtype, device=device)

    import time
    start = time.time()
    for _ in xrange(10000):
        e, grad = potTF.getEnergyGradient(weights)
    end = time.time()
    print "time", end - start
    print 'e:{0:.15f}, norm(g):{0:.15f}'.format(e, np.linalg.norm(grad))

    # results = minimize(potTF, weights)
    # print potTF.test_model(np.array(results.coords), teX, teY)

    # from pele.transition_states import findLowestEigenVector, analyticalLowestEigenvalue
    # res = findLowestEigenVector(np.array(results.coords), potTF, orthogZeroEigs=None, iprint=1, tol=1e-6)
    # print res.H0
    # print analyticalLowestEigenvalue(np.array(results.coords), potTF)

if __name__ == "__main__":
    main()