from __future__ import division
import numpy as np
import tensorflow as tf
from learnscapes.utils import *
from pele.optimize import Result


class GradientDescent(tf.train.GradientDescentOptimizer):
    def __init__(self, potential, learning_rate=1, use_locking=False, tol=1e-5, nsteps=1e3, iprint=-1):
        super(GradientDescent, self).__init__(float(learning_rate), use_locking=use_locking, name='Adagrad')
        self.debug = False
        self.nsteps = int(nsteps)
        self.iprint = self.nsteps if iprint < 0 else int(iprint)
        self.start_lr = self._learning_rate
        self.potential = potential
        self.dtype = self.potential.dtype
        self.loss = self.potential.loss
        self.x = self.potential.x
        self.x_old = tf.Variable(self.x, trainable=False)
        self.nfev = tf.Variable(tf.constant(0), tf.int32, name='nfev')
        self.rms = tf.Variable(tf.constant(1, dtype=self.dtype), name='rms')
        self.ndim = tf.constant(self.potential.nspins, dtype=self.dtype, name='ndim')
        self.sqrt_ndim = tf.constant(np.sqrt(self.potential.nspins), dtype=self.dtype, name='sqrt_ndim')
        self.tol = tf.constant(tol, dtype=self.dtype, name='tol')
        self.success = tf.Variable(False, tf.bool, name='success')
        #this initializes the graph
        self.train_op = self.minimize(self.loss)
        self.session = self.potential.session
        init = tf.initialize_all_variables()
        self.session.run(init)

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
        self.session.run(self.x.assign(coords))

    def stop_criterion_satisfied(self, grad):
        return tf.less_equal(self.rms, self.tol)

    def one_iteration(self):
        grad, coords = self.compute_gradients(self.loss, var_list=[self.x])[0]
        grad, coords = self.process_grad(grad, coords)
        return tf.group(self.rms.assign(self.compute_rms(grad)),
                        self.success.assign(self.stop_criterion_satisfied(grad)),
                        self.apply_gradients([(grad, coords)], global_step=self.nfev),
                        self.x.assign(Nnorm(self.x, self.sqrt_ndim)))

    def run(self, coords, niter=1000, decay_rate=0.):
        self.set_coords(coords)
        for i in xrange(niter):
            self.session.run(self.one_iteration())
            self._learning_rate /= (decay_rate*i+1)
            if i % self.iprint == 0:
                if self.debug:
                    assert all(self.x.eval(session=self.session) == self.potential.x.eval(session=self.session))
                success, rms, e = self.session.run([self.success, self.rms, self.loss])
                print "step: {}, e: {}, rms: {}, success: {}".format(i, e, rms, success)
                if success:
                    break

    def get_results(self):
        res = Result()
        with self.session.as_default():
            g, x = self.compute_gradients(self.loss, var_list=[self.x])[0]
            g, x = self.process_grad(g, x)
            res.energy = self.loss.eval()
            res.coords = x.eval()
            res.grad = g.eval()
            res.nfev = self.nfev.eval()
            res.rms = self.rms.eval()
            res.success = self.success.eval()
            res.nsteps = self.neval
        return res