from __future__ import division
import tensorflow as tf

def tfDot(x, y):
    return tf.reduce_sum(tf.mul(x,y))

def tfNorm(x):
    return tf.sqrt(tf.reduce_sum(tf.square(x)))

def tfRnorm(x):
    return tf.rsqrt(tf.reduce_sum(tf.square(x)))

def tfOrthog(x, zerov):
    ox = tf.sub(x, tf.mul(tfDot(x, zerov), zerov))
    return tf.sub(ox, tf.mul(tfDot(ox, zerov), zerov))

def tfNnorm(x, sqrtN):
    return tf.mul(x, tf.mul(tfRnorm(x), sqrtN))

def reduce_mean(tensor):
    den = tf.cast(tf.size(tensor), tensor.dtype)
    return tf.truediv(tf.reduce_sum(tensor), den)

def l2_loss(tensor):
    # return tf.truediv(tf.reduce_sum(tf.square(tensor)), 2.)
    return tf.nn.l2_loss(tensor)