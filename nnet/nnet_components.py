from __future__ import print_function, absolute_import
from main import tf
from main import np
from main import range
from .initializers import he_weights_initializer, identity_weights_initializer
from .misc import print_tensor_shape

# ==============================================================================
#                                                                     LEAKY_RELU
# ==============================================================================
def leaky_relu(x, rate=0.01, name="leaky_relu"):
    """Leaky Rectified Linear Activation Unit
    
    Args:
        x:    preactivation tensor
        rate: Amount of leakiness
        name: name for this op
    """
    with tf.name_scope(name) as scope:
        leak_rate = tf.mul(x, rate, name="leak_rate")
        activation = tf.maximum(x, leak_rate, name=scope)
        # activation_summary(activation)
        # tf.histogram_summary(scope + '/activation', activation)
    return activation


# ==============================================================================
#                                                                       FC_LAYER
# ==============================================================================
def fc_layer(x, n=32, bias=0.1, winit=he_weights_initializer(), dtype=tf.float32, name="fc_layer"):
    """Fully Connected Layer
    Args:
        x:          The tensor from the previous layer.
        n:          The number of nodes for this layer (width).
        bias:       (float or None)(default=0.1)
                    Initial value for bias units.
                    If None, then no Biases will be used.
        winit:      weights initializer
        name:       Name for this operation
    """
    in_nodes = int(x.get_shape()[-1]) # The number of nodes from input x
    with tf.name_scope(name) as scope:
        weights = tf.Variable(winit([in_nodes, n]),
                              name="weights", dtype=dtype)
        if bias is not None:
            bias = tf.Variable(tf.constant(bias, shape=[n], dtype=dtype), name="bias")
            preactivation = tf.add(tf.matmul(x, weights), bias, name=scope)
        else:
            preactivation = tf.matmul(x, weights, name=scope)
    return preactivation


# ==============================================================================
#                                                                        FLATTEN
# ==============================================================================
def flatten(x, name="flatten"):
    """Given a tensor whose first dimension is the batch_size, and all other
    dimensions are elements of data, then it flattens the data so you get a
    [batch_size, num_elements] sized tensor"""
    with tf.name_scope(name) as scope:
        num_elements = np.product(x.get_shape().as_list()[1:])
        x = tf.reshape(x, [-1, num_elements], name=scope)
    return x


# ==============================================================================
#                                                                     CONV_LAYER
# ==============================================================================
def conv_layer(x, k=3, n=8, stride=1, bias=0.1,
               winit=he_weights_initializer(), pad='SAME', name="conv_layer"):
    num_filters_in = int(x.get_shape()[-1])  # The number of chanels coming in
    
    with tf.name_scope(name) as scope:
        weights = tf.Variable(winit([k, k, num_filters_in, n]), name="weights",
                              dtype=tf.float32)
        if bias is not None:
            conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1],
                                padding=pad, name="conv")
            bias = tf.Variable(tf.constant(bias, shape=[n]), name="bias")
            preactivation = tf.add(conv, bias, name=scope)
        else:
            preactivation = tf.nn.conv2d(x, weights,
                                         strides=[1, stride, stride, 1],
                                         padding=pad, name=scope)
    
    return preactivation


# ==============================================================================
#                                                                 MAX_POOL_LAYER
# ==============================================================================
def max_pool_layer(x, k=2, stride=2, name="maxpool"):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                          strides=[1, stride, stride, 1], padding='SAME',
                          name=name)


# ==============================================================================
#                                                                 AVG_POOL_LAYER
# ==============================================================================
def avg_pool_layer(x, k=2, stride=2, name="avg_pool"):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1],
                          padding="SAME", name=name)


# ==============================================================================
#                                                                      BATCHNORM
# ==============================================================================
def batchnormC(x, is_training, iteration, conv=False, offset=0.0, scale=1.0):
    """
    Given some logits `x`, apply batch normalization to them.

    Parameters
    ----------
    x
    is_training
    iteration
    conv:      (boolean)(default=False)
        Applying it to a convolutional layer?

    Returns
    -------


    Credits
    -------
    This code is based on code written by Martin Gorner:
    - https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.2_batchnorm_convolutional.py
    https://www.youtube.com/watch?v=vq2nnJ4g6N0
    """
    # adding the iteration prevents from averaging across non-existing iterations
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.9999, iteration)
    bnepsilon = 1e-5
    
    # calculate mean and variance for batch of logits
    if conv:
        mean, variance = tf.nn.moments(x, [0, 1, 2])
    else:
        # mean and variance along the batch
        mean, variance = tf.nn.moments(x, [0])
    
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    tf.add_to_collection("update_moving_averages", update_moving_averages)
    
    # Mean and Variance (how it get it is dependent on whether it is training)
    # TODO: Change the following to use the `is_trianing` directly without logical_not()
    #       to make it more intuitive.
    m = tf.cond(tf.logical_not(is_training),
                lambda: exp_moving_avg.average(mean),
                lambda: mean)
    v = tf.cond(tf.logical_not(is_training),
                lambda: exp_moving_avg.average(variance),
                lambda: variance)
    
    # Offset
    param_shape = mean.get_shape().as_list()
    beta_init = tf.constant_initializer(offset)
    beta = tf.Variable(initial_value=beta_init(param_shape), name="beta")
    
    # Scale
    gamma_init = tf.constant_initializer(scale)
    gamma = tf.Variable(initial_value=gamma_init(param_shape), name="gamma")
    
    # Apply Batch Norm
    Ybn = tf.nn.batch_normalization(x, m, v, offset=beta, scale=gamma,
                                    variance_epsilon=bnepsilon)
    return Ybn



# ==============================================================================
#                                                                   CONV_BATTERY
# ==============================================================================
def conv_battery(x, global_step, convk=3, n=32, mpk=2, mpstride=1,
                 dropout=0.0, is_training=False, name="C", verbose=False):
    # BATCH NORM SETTINGS
    bn_offset = 0.0
    bn_scale = 1.0
    
    # DROPOUT
    conv_dropout = tf.cond(is_training,
                           lambda: tf.constant(dropout),
                           lambda: tf.constant(0.0))
    # CONV STACK
    with tf.name_scope(name) as scope:
        x = conv_layer(x, k=convk, n=n, bias=None, stride=1, name="conv")
        x = batchnormC(x, is_training=is_training,
                       iteration=global_step, conv=True,
                       offset=bn_offset, scale=bn_scale)
        x = tf.nn.dropout(x, keep_prob=1 - conv_dropout)
        x = max_pool_layer(x, k=mpk, stride=mpstride, name="maxpool")
        x = leaky_relu(x, name="relu")
        print_tensor_shape(x, name=scope, verbose=verbose)
    
    return x


# ==============================================================================
#                                                                     FC_BATTERY
# ==============================================================================
def fc_battery(x, global_step, n=1024, bias=0.1, is_training=False,
               dropout=0.0, winit=None, verbose=False, name="FC"):
    # BATCH NORM SETTINGS
    bn_offset = 0.0
    bn_scale = 1.0
    
    # DROPOUT
    dropout = tf.cond(is_training,
                      lambda: tf.constant(dropout),
                      lambda: tf.constant(0.0))
    
    # DEFAULT WEIGHTS INITIALIZATION
    if winit is None:
        winit = he_weights_initializer()  # identity_weights_initializer()
    
    # FC STACK
    with tf.name_scope(name) as scope:
        x = fc_layer(x, n=n, bias=bias, winit=winit, name="FC")
        x = batchnormC(x,
                       is_training=is_training,
                       iteration=global_step,
                       conv=False,
                       offset=bn_offset,
                       scale=bn_scale)
        x = tf.nn.dropout(x, keep_prob=1 - dropout)
        x = leaky_relu(x, rate=0.01, name="relu")
        print_tensor_shape(x, name=scope, verbose=verbose)
    
    return x


# ==============================================================================
#                                                                        TRAINER
# ==============================================================================
def trainer(loss, alpha=0.001, global_step=None, name="train"):
    """ Performs Adam Optimization and Goes through a learning step given
        a loss (and optionally alpha and global step)

    Args:
        loss:           (float) Loss value
        alpha:          (float or tensor) The alpha learning rate to use.
        global_step:    (tensor) the tensor storing the global step.
        name:           (str) Name for the scope of this training op.

    Returns:
        (op) the minimize operation performed.
    """
    # with tf.name_scope(name) as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=0.9, beta2=0.999,
                                       epsilon=1e-08,
                                       name="optimizer")
    train = optimizer.minimize(loss, global_step=global_step, name=name)
    return train

