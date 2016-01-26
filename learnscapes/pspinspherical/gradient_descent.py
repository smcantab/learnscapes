from __future__ import division
import numpy as np
import tensorflow as tf
from learnscapes.utils import *
from pele.optimize import Result


class GradientDescent(tf.train.GradientDescentOptimizer):
    def __init__(self, model, learning_rate=1, use_locking=False, tol=1e-5, nsteps=1e3, iprint=-1, device='gpu'):
        super(GradientDescent, self).__init__(float(learning_rate), use_locking=use_locking, name='Adagrad')
        self.debug = False
        self.nsteps = int(nsteps)
        self.iprint = self.nsteps if iprint < 0 else int(iprint)
        self.start_lr = self._learning_rate
        self.device = select_device_simple(dev=device)
        self.dtype = model.dtype
        # the following scheme needs to be followed to avoid repeated addition of ops to
        # the graph
        self.g = tf.Graph()
        with self.g.as_default(), self.g.device(self.device):
            self.session = tf.Session(config=tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False))
            with self.session.as_default():
                self.model = model(graph=self.g)
                with self.g.name_scope("minimizer_embedding"):
                    #with self.g.name_scope("minimizer"):
                    self.nfev = tf.Variable(tf.constant(0), tf.int32, name='nfev')
                    self.rms = tf.Variable(tf.constant(1, dtype=self.dtype), name='rms')
                    self.ndim = tf.constant(self.model.nspins, dtype=self.dtype, name='ndim')
                    self.sqrt_ndim = tf.constant(np.sqrt(self.model.nspins), dtype=self.dtype, name='sqrt_ndim')
                    self.tol = tf.constant(tol, dtype=self.dtype, name='tol')
                    self.success = tf.Variable(False, tf.bool, name='success')
                    #this initializes the graph
                with self.g.name_scope("minimizer_ops"):
                    self.train_op = self.minimize(self.model.gloss)
                    self.one_iteration_op = self.one_iteration
                init = tf.initialize_all_variables()
                self.session.run(init)
                self.g.finalize()   # this guarantees that no new ops are added to the graph

    def _valid_dtypes(self):
        return set([tf.float32, tf.float64])

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()

    def process_grad(self, grad, coords):
        """
        only grad can be modified because coords needs to remain a variable
        if modified it's no longer a variable
        :param grad:
        :param coords:
        :return:
        """
        zerov = tf.mul(coords, tfRnorm(coords))
        grad = tfOrthog(grad, zerov)
        return grad, coords

    def compute_rms(self, grad):
        return tf.div(tfNorm(grad), self.sqrt_ndim)

    def set_coords(self, coords):
        return self.model.x.assign(coords)

    def stop_criterion_satisfied(self, grad):
        return tf.less_equal(self.rms, self.tol)

    def one_iteration(self):

        grad, coords = self.compute_gradients(self.model.gloss, var_list=[self.model.x])[0]
        grad, coords = self.process_grad(grad, coords)
        return tf.group(self.rms.assign(self.compute_rms(grad)),
                        self.success.assign(self.stop_criterion_satisfied(grad)),
                        self.apply_gradients([(grad, coords)], global_step=self.nfev),
                        self.model.x.assign(Nnorm(self.model.x, self.sqrt_ndim)))

    def run(self, coords, niter=1000, decay_rate=0.):
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                # self.session.run(self.model.x.assign(coords))
                for i in xrange(niter):
                    self.session.run(self.one_iteration_op())
                    self._learning_rate /= (decay_rate*i+1)
                    if i % self.iprint == 0:
                        if self.debug:
                            assert all(self.model.x.eval() == self.model.x.eval())
                        success, rms, e = self.session.run([self.success, self.rms, self.model.gloss])
                        print "step: {}, e: {}, rms: {}, success: {}".format(i, e, rms, success)
                        if success:
                            break

    def get_results(self):
        res = Result()
        with self.g.as_default(), self.g.device(self.device):
            with self.session.as_default():
                g, x = self.compute_gradients(self.model.gloss, var_list=[self.model.x])[0]
                g, x = self.process_grad(g, x)
                res.energy = self.model.gloss.eval()
                res.coords = x.eval()
                res.grad = g.eval()
                res.nfev = self.nfev.eval()
                res.rms = self.rms.eval()
                res.success = self.success.eval()
                res.nsteps = self.neval
        return res