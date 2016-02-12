from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pele.potentials import BasePotential
from pele.optimize._quench import lbfgs_cpp
from learnscapes.NeuralNets import BaseMLGraph
from learnscapes.utils import select_device_simple


class LogisticRegressionGraph(BaseMLGraph):
    """
    linear regression model of the form y = w*x, so this model is very simple
    """
    def __init__(self, x_train_data, y_train_data, reg=0, dtype='float32'):
        super(LogisticRegressionGraph, self).__init__(x_train_data, y_train_data, reg=reg,dtype=dtype)

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
            self.w = tf.Variable(tf.zeros(self.shape, dtype=self.dtype), name='weights')
            self.b = tf.Variable(tf.zeros([self.shape[1]], dtype=self.dtype), name='bias')
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
        return tf.matmul(self.x, self.w) + self.b

    @property
    def regularization(self):
        return self.reg * tf.nn.l2_loss(self.w)

    @property
    def compute_gradient(self):
        return tf.gradients(self.regularised_loss, [self.w, self.b])

    @property
    def predict(self):
        """this tests the models, at predict time, evaluate the argmax of the logistic regression
        """
        model_test = tf.matmul(self.x_test, self.w) + self.b
        return tf.argmax(model_test, 1)


class LogisticRegressionPotential(BasePotential):
    """
    potential that can be used with pele toolbox
    """
    def __init__(self, x_train_data, y_train_data, reg=0, dtype='float32', device='cpu'):
        self.device = select_device_simple(dev=device)
        # the following scheme needs to be followed to avoid repeated addition of ops to
        # the graph
        self.g = tf.Graph()
        with self.g.as_default(), self.g.device(self.device):
            self.session = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False))
            with self.session.as_default():
                self.tf_graph = LogisticRegressionGraph(x_train_data, y_train_data,
                                                        reg=reg, dtype=dtype)(graph=self.g)
                init = tf.initialize_all_variables()
                self.session.run(init)
                self.g.finalize()   # this guarantees that no new ops are added to the graph

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

    def pele_to_tf(self, coords):
        featdim, outdim = self.tf_graph.shape
        w = coords[:featdim*outdim].reshape((featdim, outdim))
        b = coords[-outdim:]
        return w, b

    def tf_to_pele(self, w, b):
        return np.append(w.flatten(), b)

    def getEnergy(self, coords):
        w, b = self.pele_to_tf(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                e = self.session.run(self.tf_graph.gloss, feed_dict={self.tf_graph.w: w, self.tf_graph.b: b})
        return e

    def getEnergyGradient(self, coords):
        w, b = self.pele_to_tf(coords)
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                operations = [self.tf_graph.gloss] + self.tf_graph.gcompute_gradient #extend operations
                e, grad_w, grad_b = self.session.run(operations,
                                                     feed_dict={self.tf_graph.w: w, self.tf_graph.b: b})
        return e, self.tf_to_pele(grad_w, grad_b)

    def test_model(self, coords, x_test, y_test):
        w, b = self.pele_to_tf(coords)
        prediction = self.session.run([self.tf_graph.gpredict],
                                      feed_dict={self.tf_graph.w: w,
                                                 self.tf_graph.b: b,
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

    dtype = 'float32'
    device = 'gpu'

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    bs = -1
    trX, trY, teX, teY = mnist.train.images[:bs], mnist.train.labels[:bs], mnist.test.images, mnist.test.labels

    # like in linear regression, we need a shared variable weight matrix for logistic regression
    w = init_weights([784, 10])
    b = np.ones(10)*0.1
    weights = np.append(w, b)
    potTF = LogisticRegressionPotential(trX, trY, reg=0.01, dtype=dtype, device=device)

    e, grad = potTF.getEnergyGradient(weights)
    print 'e:{0:.15f}, norm(g):{0:.15f}'.format(e, np.linalg.norm(grad))

    results = minimize(potTF, weights)
    print potTF.test_model(np.array(results.coords), teX, teY)

if __name__ == "__main__":
    main()