from __future__ import print_function, absolute_import

# ==============================================================================
#                                                             PRINT_TENSOR_SHAPE
# ==============================================================================
def print_tensor_shape(x, name=None, verbose=True):
    """ Prints out the shape of the specified tensor.
        By default it uses the name actually assigned to the tensor by
        Tensroflow, but if you provide the optional argument `name` it
        uses that as the name that gets printed out.
    Args:
        x:       (Tensorflow tensor)
        name:    (str or None) Optional alternative name used in printout.
        verbose: (bool)(default=True)
                 If set to False, it prints out nothing.
                 Seems pointless, but it can make your code prettier by avoiding
                 lots of `if` statements for when you only want to print this
                 information out when `verbose` flag is set to True.
    """
    if verbose:
        if name is None:
            print("-shape of {}: {}".format(x.op.name, x.get_shape().as_list()))
        else:
            print("-shape of {}: {}".format(name, x.get_shape().as_list()))


# ==============================================================================
#                                                             VARIABLE_SUMMARIES
# ==============================================================================
def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor.
    Code taken from:
    https://www.tensorflow.org/versions/r0.11/how_tos/summaries_and_tensorboard/
    """
    pass
    # with tf.name_scope('summaries'):
    #     mean = tf.reduce_mean(var)
    #     tf.scalar_summary('mean/' + name, mean)
    #     with tf.name_scope('stddev'):
    #       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    #     tf.scalar_summary('stddev/' + name, stddev)
    #     tf.scalar_summary('max/' + name, tf.reduce_max(var))
    #     tf.scalar_summary('min/' + name, tf.reduce_min(var))
    #     tf.histogram_summary(name, var)

