from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from pele.potentials import BasePotential
from pele.optimize._quench import lbfgs_cpp
from learnscapes.NeuralNets import DoubleLogisticRegressionGraph, DoubleLogisticRegressionPotential
from learnscapes.utils import select_device_simple

# this class is identical to the DoubleLogistic regression but uses exponential activiation units instead
# of sigmoids as non-linearities

class Elu2NNGraph(DoubleLogisticRegressionGraph):
    """
    this is a 2-layer modern NN that uses exponential activation functions
    :param hnodes: number of hidden nodes
    """
    def __init__(self, x_train_data, y_train_data, hnodes, reg=0, dtype='float32'):
        super(Elu2NNGraph, self).__init__(x_train_data, y_train_data, hnodes, reg=reg,dtype=dtype)

    @property
    def model(self):
        h = tf.nn.elu(tf.add(tf.matmul(self.x, self.w_h), self.b_h))
        return tf.add(tf.matmul(h, self.w_o), self.b_o)

    @property
    def predict(self):
        """this tests the models, at predict time, evaluate the argmax of the logistic regression
        """
        h = tf.nn.elu(tf.add(tf.matmul(self.x_test, self.w_h), self.b_h))
        model_test = tf.add(tf.matmul(h, self.w_o), self.b_o)
        return tf.argmax(model_test, 1)


class Elu2NNPotential(DoubleLogisticRegressionPotential):
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
                self.tf_graph = Elu2NNGraph(x_train_data, y_train_data,
                                            hnodes, reg=reg, dtype=dtype)(graph=self.g)
                init = tf.initialize_all_variables()
                self.session.run(init)
                self.g.finalize()   # this guarantees that no new ops are added to the graph
        self.ndim = self.tf_graph.ndim

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
    device = 'cpu'

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    bs = 1000
    trX, trY, teX, teY = mnist.train.images[:bs], mnist.train.labels[:bs], mnist.test.images, mnist.test.labels

    # like in linear regression, we need a shared variable weight matrix for logistic regression
    hnodes = 100 #625
    w_h = init_weights([784, hnodes])
    w_o = init_weights([hnodes, 10])
    w = np.append(w_h, w_o)
    b_h = np.ones(hnodes)*0.1
    b_o = np.ones(10)*0.1
    b = np.append(b_h, b_o)
    weights = np.append(w, b)
    potTF = Elu2NNPotential(trX, trY, hnodes, dtype=dtype, device=device)

    e, grad = potTF.getEnergyGradient(weights)
    print 'e:{0:.15f}, norm(g):{0:.15f}'.format(e, np.linalg.norm(grad))

    results = minimize(potTF, weights)
    print potTF.test_model(np.array(results.coords), teX, teY)

    # from pele.transition_states import findLowestEigenVector, analyticalLowestEigenvalue
    # res = findLowestEigenVector(np.array(results.coords), potTF, orthogZeroEigs=None, iprint=1, tol=1e-6)
    # print res.H0
    # print analyticalLowestEigenvalue(np.array(results.coords), potTF)

if __name__ == "__main__":
    main()