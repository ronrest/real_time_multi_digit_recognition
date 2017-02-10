from __future__ import print_function, absolute_import
from main import tf
from main import np

# ==============================================================================
#                                                              HE_NORMAL_WEIGHTS
# ==============================================================================
def he_weights_initializer(dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        # print("initializing with HE_WEIGHTS")
        shape = list(shape)
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        sd = np.sqrt(2. / fan_in)
        return tf.truncated_normal(shape, stddev=sd, dtype=dtype)

    return _initializer


# ==============================================================================
#                                                                         GLOROT
# ==============================================================================
def glorot_initializer(dtype=tf.float32, scale=6.0, uniform=False, seed=None):
    def _initializer(shape, dtype=dtype, partition_info=None):
        #print("initializing with GLOROT_UNIFORM WEIGHTS")
        #     return uniform(shape, s, name=name)
        
        shape = list(shape)
        receptive_field_size = np.prod(shape[:2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1]
        sd = np.sqrt(float(scale) / (fan_in + fan_out))
        if uniform:
            return tf.random_uniform(shape, minval=-sd, maxval=sd, dtype=dtype, seed=seed)
        else:
            return tf.truncated_normal(shape, stddev=sd, dtype=dtype, seed=seed)
    return _initializer


# ==============================================================================
#                                                                 XAVIER_WEIGHTS
# ==============================================================================
def xavier_weights(shape, name="weights"):
    # print("creating XAVIER_WEIGHTS")
    shape = list(shape)
    receptive_field_size = np.prod(shape[:2])
    fanin = shape[-2] # * receptive_field_size
    fanout = shape[-1]
    # print("WEIGHTS: shape: {} Fanin {} fanout {} receptive field {}".format(shape, fanin, fanout, receptive_field_size))
    #W = np.random.randn(fanin, fanout) / np.sqrt(fanin)
    W = tf.truncated_normal(shape) / np.sqrt(fanin)
    return tf.Variable(W, dtype=tf.float32, name=name)


# ==============================================================================
#                                                                    STD_WEIGHTS
# ==============================================================================
def std_weights(shape, std=0.01, name="weights"):
    # print("creating STD_WEIGHTS")
    shape = list(shape)
    return tf.Variable(tf.truncated_normal(shape, stddev=std), name=name)



# ==============================================================================
#                                                                   ZERO_WEIGHTS
# ==============================================================================
def zero_weights_initializer(dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        # print("initializing with ZERO_WEIGHTS")
        shape = list(shape)
        return tf.zeros(shape, dtype=dtype)
    return _initializer


# ==============================================================================
#                                                               IDENTITY_WEIGHTS
# ==============================================================================
def identity_weights_initializer(dtype=tf.float32):
    """
    NOTE: This is fo FC layers, it will NOT work for convolutional layer weights
    """
    def _initializer(shape, dtype=dtype, partition_info=None):
        # print("initializing with IDENTITY_WEIGHTS")
        shape = list(shape)
        return tf.constant(np.eye(shape[0], shape[1]), dtype=dtype)
    return _initializer


# ==============================================================================
#                                                         NOISY_IDENTITY_WEIGHTS
# ==============================================================================
def noisy_identity_weights(shape, noise=0.0001, name="weights"):
    #return he_normal_weights(shape, name="name")
    # print("creating NOISY_IDENTITY_WEIGHTS")
    shape = list(shape)
    noise = noise*np.random.randn(*shape)
    return tf.Variable(noise + np.eye(*shape), dtype=tf.float32, name=name)

