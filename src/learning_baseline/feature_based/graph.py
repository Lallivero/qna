from collections import namedtuple
import tensorflow as tf

Variables = namedtuple('Variables', ['W'])

def GetVariables(feature_counter):
    W = tf.Variable(tf.zeros([feature_counter.NumFeatures()]), name='W')
    return Variables(W)

def GetLogits(inputs, variables):
    return tf.reduce_sum(inputs.weight_scaling_constant * tf.gather(variables.W, inputs.input_indices) * tf.to_float(inputs.input_mask), 2)
