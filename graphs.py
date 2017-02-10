from __future__ import print_function, absolute_import, division
from main import tf
from main import PRINT_WIDTH
from support import print_headers

# Neural Net Imports
from nnet.graphops import GraphOps
from nnet.initializers import he_weights_initializer
from nnet.model_components import multi_digit_loss
from nnet.nnet_components import fc_layer, flatten
from nnet.nnet_components import conv_battery, fc_battery
from nnet.nnet_components import trainer
from nnet.misc import print_tensor_shape


# ##############################################################################
#                                                                          GRAPH
# ##############################################################################
def create_graph(logit_func, settings):
    """ Creates a Tensorflow graph for the multi-digit classification + bounding
        box task.
        
    Args:
        logit_func: (function) A function that returns two tensors:
                    - digit_logits
                    - bbox_logits
        settings:   (object) A Settings object that contains attribute values
                    for the model.
    Returns:
        (tensorflow graph)
    """
    print_headers("GRAPH", border="=", width=PRINT_WIDTH)
    graph = tf.Graph()
    with graph.as_default():
        # PLACEHOLDERS
        X = tf.placeholder(tf.float32, shape=[None, 54, 54], name="X")  # Images
        Y = tf.placeholder(tf.int32, shape=[None, 5], name="Y")         # Digits
        BBOX = tf.placeholder(tf.float32, shape=[None, 24], name="BBOX")# Bboxes
        
        # OPTIONAL PLACEHOLDERS
        alpha = tf.placeholder_with_default(0.001, shape=None, name="alpha")
        is_training = tf.placeholder_with_default(False,
                                                  shape=None,
                                                  name="is_training")
        
        # VARIABLES
        global_step = tf.Variable(0, name='global_step', trainable=False)
        
        # PREPROCESS
        x = X / 255.0  # Rescale values to be 0-1
        x = tf.reshape(x, shape=[-1, 54, 54, 1])  # Reshape for Conv Layers
        print("x after reshaping to 4D: ", x.get_shape().as_list())
        
        # MODEL
        digit_logits, bbox_logits = logit_func(x=x,
                                               is_training=is_training,
                                               settings=settings,
                                               global_step=global_step)
        
        # BBOX LOSS
        bbox_loss = tf.sqrt(tf.reduce_mean(tf.square(1 * (bbox_logits - BBOX))),
                            name="bbox_loss")
        
        # DIGITS LOSS
        digits_loss = multi_digit_loss(digit_logits, Y,
                                       max_digits=5,
                                       name="digit_loss")
        
        # TOTAL LOSS
        loss = tf.add(bbox_loss, digits_loss, name="loss")
        
        # TRAIN
        train = trainer(loss, alpha=alpha, global_step=global_step,name="train")
        
        # PREDICTIONS
        digit_preds = tf.transpose(tf.argmax(digit_logits, dimension=2))
        digit_preds = tf.to_int32(digit_preds, name="digit_preds")
    
    return graph


# ==============================================================================
#                                                                        MODEL_A
# ==============================================================================
def model_a(x, is_training, global_step, settings=None, verbose=True):
    """
    """
    # BATCH NORM SETTINGS
    bn_offset = 0.0
    bn_scale = 1.0
    
    # MISC SETTINGS
    bval = 0.01  # Bias value
    leak = 0.01  # leakiness of leaky relus
    
    # WEIGHTS INITIALIZERS
    # st_winit = zero_weights_initializer()
    conv_winit = he_weights_initializer()
    fc_winit = he_weights_initializer()  # identity_weights_initializer()
    
    # DROPOUT SETTINGS
    conv_dropout = tf.cond(is_training,
                           lambda: tf.constant(settings.conv_dropout),
                           lambda: tf.constant(0.0))
    fc_dropout = tf.cond(is_training,
                         lambda: tf.constant(settings.fc_dropout),
                         lambda: tf.constant(0.0))
    
    # --------------------------------------------------------------------------
    #                                                                      TRUNK
    # --------------------------------------------------------------------------
    # CONV LAYERS
    x = conv_battery(x, global_step=global_step, convk=5, n=48, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=64, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=128, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=160, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=2, n=192, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    # FC LAYER
    x = flatten(x)
    print_tensor_shape(x, verbose=verbose)
    x = fc_battery(x, global_step=global_step, n=1024, bias=None,
                   is_training=is_training, dropout=settings.fc_dropout,
                   winit=fc_winit, verbose=verbose, name="FC")
    
    # --------------------------------------------------------------------------
    #                                                             DIGIT BRANCHES
    # --------------------------------------------------------------------------
    max_digits = 5
    d = [None] * max_digits
    for i in range(max_digits):
        d[i] = fc_layer(x, n=11, bias=0.1, winit=fc_winit,
                        name="branch_{}".format(i + 1))
        print_tensor_shape(d[i], verbose=verbose)
    
    digits = tf.pack(d, axis=0, name="digit_logits")
    print_tensor_shape(digits, verbose=verbose)
    
    # --------------------------------------------------------------------------
    #                                                              BBOX BRANCHES
    # --------------------------------------------------------------------------
    bboxes = fc_layer(x, n=24, bias=0.1, winit=fc_winit, name="bbox_logits")
    print_tensor_shape(bboxes, verbose=verbose)
    
    return digits, bboxes


# ==============================================================================
#                                                                        MODEL_B
# ==============================================================================
# Similar to model A, but has aditional conv layer at begining with:
# - k=2, n = 2
# - maxpool k =3, stride=2
# This is intended to reduce the dimensionality early on, while preserving
# important information.
def model_b(x, is_training, global_step, settings=None, verbose=True):
    """
    """
    # BATCH NORM SETTINGS
    bn_offset = 0.0
    bn_scale = 1.0
    
    # MISC SETTINGS
    bval = 0.01  # Bias value
    leak = 0.01  # leakiness of leaky relus
    
    # WEIGHTS INITIALIZERS
    # st_winit = zero_weights_initializer()
    conv_winit = he_weights_initializer()
    fc_winit = he_weights_initializer()  # identity_weights_initializer()
    
    # DROPOUT SETTINGS
    conv_dropout = tf.cond(is_training,
                           lambda: tf.constant(settings.conv_dropout),
                           lambda: tf.constant(0.0))
    fc_dropout = tf.cond(is_training,
                         lambda: tf.constant(settings.fc_dropout),
                         lambda: tf.constant(0.0))
    
    # --------------------------------------------------------------------------
    #                                                                      TRUNK
    # --------------------------------------------------------------------------
    # CONV LAYERS
    x = conv_battery(x, global_step=global_step, convk=2, n=2, mpk=3,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=48, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=64, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=128, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=160, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=2, n=192, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    # FC LAYER
    x = flatten(x)
    print_tensor_shape(x, verbose=verbose)
    x = fc_battery(x, global_step=global_step, n=1024, bias=None,
                   is_training=is_training, dropout=settings.fc_dropout,
                   winit=fc_winit, verbose=verbose, name="FC")
    
    # --------------------------------------------------------------------------
    #                                                             DIGIT BRANCHES
    # --------------------------------------------------------------------------
    max_digits = 5
    d = [None] * max_digits
    for i in range(max_digits):
        d[i] = fc_layer(x, n=11, bias=0.1, winit=fc_winit,
                        name="branch_{}".format(i + 1))
        print_tensor_shape(d[i], verbose=verbose)
    
    digits = tf.pack(d, axis=0, name="digit_logits")
    print_tensor_shape(digits, verbose=verbose)
    
    # --------------------------------------------------------------------------
    #                                                              BBOX BRANCHES
    # --------------------------------------------------------------------------
    bboxes = fc_layer(x, n=24, bias=0.1, winit=fc_winit, name="bbox_logits")
    print_tensor_shape(bboxes, verbose=verbose)
    
    return digits, bboxes


# ==============================================================================
#                                                                        MODEL_C
# ==============================================================================
# Similar to model A, but has aditional conv layer at begining with:
# - k=2, n = 16
# - maxpool k =2, stride=2
# This is intended to reduce the dimensionality early on, while preserving
# important information.
def model_c(x, is_training, global_step, settings=None, verbose=True):
    """
    """
    # BATCH NORM SETTINGS
    bn_offset = 0.0
    bn_scale = 1.0
    
    # MISC SETTINGS
    bval = 0.01  # Bias value
    leak = 0.01  # leakiness of leaky relus
    
    # WEIGHTS INITIALIZERS
    # st_winit = zero_weights_initializer()
    conv_winit = he_weights_initializer()
    fc_winit = he_weights_initializer()  # identity_weights_initializer()
    
    # DROPOUT SETTINGS
    conv_dropout = tf.cond(is_training,
                           lambda: tf.constant(settings.conv_dropout),
                           lambda: tf.constant(0.0))
    fc_dropout = tf.cond(is_training,
                         lambda: tf.constant(settings.fc_dropout),
                         lambda: tf.constant(0.0))
    
    # --------------------------------------------------------------------------
    #                                                                      TRUNK
    # --------------------------------------------------------------------------
    # CONV LAYERS
    x = conv_battery(x, global_step=global_step, convk=2, n=16, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=48, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=64, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=128, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=160, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=2, n=192, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    # FC LAYER
    x = flatten(x)
    print_tensor_shape(x, verbose=verbose)
    x = fc_battery(x, global_step=global_step, n=1024, bias=None,
                   is_training=is_training, dropout=settings.fc_dropout,
                   winit=fc_winit, verbose=verbose, name="FC")
    
    # --------------------------------------------------------------------------
    #                                                             DIGIT BRANCHES
    # --------------------------------------------------------------------------
    max_digits = 5
    d = [None] * max_digits
    for i in range(max_digits):
        d[i] = fc_layer(x, n=11, bias=0.1, winit=fc_winit,
                        name="branch_{}".format(i + 1))
        print_tensor_shape(d[i], verbose=verbose)
    
    digits = tf.pack(d, axis=0, name="digit_logits")
    print_tensor_shape(digits, verbose=verbose)
    
    # --------------------------------------------------------------------------
    #                                                              BBOX BRANCHES
    # --------------------------------------------------------------------------
    bboxes = fc_layer(x, n=24, bias=0.1, winit=fc_winit, name="bbox_logits")
    print_tensor_shape(bboxes, verbose=verbose)
    
    return digits, bboxes


# ==============================================================================
#                                                                        MODEL_D
# ==============================================================================
# Similar to model A, but has aditional conv layer at begining with:
# - k=2, n = 16
# - maxpool k =2, stride=2
#
#  AND ALSO:
#  There is no fully connected layer just before the branches, it goes directly
#  from convolutional layers, to the branches.
#
# This is intended to reduce the dimensionality early on, while preserving
# important information.
def model_d(x, is_training, global_step, settings=None, verbose=True):
    """
    """
    # BATCH NORM SETTINGS
    bn_offset = 0.0
    bn_scale = 1.0
    
    # MISC SETTINGS
    bval = 0.01  # Bias value
    leak = 0.01  # leakiness of leaky relus
    
    # WEIGHTS INITIALIZERS
    # st_winit = zero_weights_initializer()
    conv_winit = he_weights_initializer()
    fc_winit = he_weights_initializer()  # identity_weights_initializer()
    
    # DROPOUT SETTINGS
    conv_dropout = tf.cond(is_training,
                           lambda: tf.constant(settings.conv_dropout),
                           lambda: tf.constant(0.0))
    fc_dropout = tf.cond(is_training,
                         lambda: tf.constant(settings.fc_dropout),
                         lambda: tf.constant(0.0))
    
    # --------------------------------------------------------------------------
    #                                                                      TRUNK
    # --------------------------------------------------------------------------
    # CONV LAYERS
    x = conv_battery(x, global_step=global_step, convk=2, n=16, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=48, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=64, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=128, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=5, n=160, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=3, n=192, mpk=2,
                     mpstride=2, is_training=is_training, verbose=verbose)
    
    x = conv_battery(x, global_step=global_step, convk=2, n=192, mpk=2,
                     mpstride=1, is_training=is_training, verbose=verbose)
    
    # FLATTEN
    x = flatten(x)
    print_tensor_shape(x, verbose=verbose)
    
    # --------------------------------------------------------------------------
    #                                                             DIGIT BRANCHES
    # --------------------------------------------------------------------------
    max_digits = 5
    d = [None] * max_digits
    for i in range(max_digits):
        d[i] = fc_layer(x, n=11, bias=0.1, winit=fc_winit,
                        name="branch_{}".format(i + 1))
        print_tensor_shape(d[i], verbose=verbose)
    
    digits = tf.pack(d, axis=0, name="digit_logits")
    print_tensor_shape(digits, verbose=verbose)
    
    # --------------------------------------------------------------------------
    #                                                              BBOX BRANCHES
    # --------------------------------------------------------------------------
    bboxes = fc_layer(x, n=24, bias=0.1, winit=fc_winit, name="bbox_logits")
    print_tensor_shape(bboxes, verbose=verbose)
    
    return digits, bboxes

