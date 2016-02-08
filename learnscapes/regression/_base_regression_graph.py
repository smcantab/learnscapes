from __future__ import division
import numpy as np
import tensorflow as tf

class BaseRegressionGraph(object):
    """
    this should not have its own graph or its own section, it defines only the graph
    """

    def __init__(self, x_train_data, y_train_data, reg=0, dtype='float32'):
        self.dtype = dtype
        self.py_x = np.array(x_train_data)
        self.py_y = np.array(y_train_data)
        self.reg = reg
        assert self.py_y.shape[0] == self.py_x.shape[0], "dataset sizes mismatch"
        self.shape = (self.py_x.shape[1], self.py_y.shape[1])

    @property
    def model(self):
        """
        this is the model, it could be linear or quadratic etc
        """
        return NotImplemented

    @property
    def loss(self):
        """this mutliplies times the interaction and reduces sum reduces the tensor
            compute mean cross entropy (softmax is applied internally)
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.model, self.y))

    @property
    def regularised_loss(self):
        return self.loss + self.regularization

    @property
    def predict(self):
        """this mutliplies times the interaction and reduces sum reduces the tensor"""
        return NotImplemented