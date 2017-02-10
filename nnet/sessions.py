from __future__ import print_function, absolute_import
from main import tf
from main import np
from main import PRINT_WIDTH
from support import assert_these_attributes, verbose_print, verbose_print_done, \
    limit_string
from .graphops import GraphOps

# ==============================================================================
#                                                   TF_INITIALIZE_VARS_FROM_FILE
# ==============================================================================
def tf_initialize_vars_from_file(f, s, saver, verbose=False):
    """Restore variables from previous checkpoint if the file exists, otherwise
    initialize to new values.

    Args:
        f:       filepath to checkpoint file
        s:       tensorflow session object
        saver:   tensorflow saver object used in the graph/session
    """
    verbose_print("Loading checkpoint from " + limit_string(f, tail=PRINT_WIDTH-37),
                  verbose=verbose, end="")

    if tf.train.checkpoint_exists(f):
        saver.restore(s, f)
    else:
        verbose_print("\n -- File does not already exist. Initializing to new values",
                      verbose=verbose, end="")
        s.run(tf.global_variables_initializer())
    
    verbose_print_done(verbose)



# ==============================================================================
#                                                                     TRAIN STEP
# ==============================================================================
def train_step(s, g, data, alpha=0.01):
    """ Performs a training step, and returns the loss

    Args:
        s:      Session
        g:      graphOps object. Must contain the following attributes:
                "X", "Y", "BBOX", "is_training", "alpha", "loss",
                "update_moving_averages"
        data:   data object containing the following attributes containing data:
                X, Y, BBOX
        alpha:  learning rate

    Returns:
        loss value
    """
    # CHECK THE GRAPHOPS OBJECT HAS ALL THE REQUIRED ATTRIBUTES.
    g_attributes = ["X", "Y", "BBOX", "is_training", "alpha", "loss",
                    "update_moving_avgs"]
    assert_these_attributes(g, name="GraphOps object", attr=g_attributes)
    
    # CHECK THE DATA OBJECTOBJECT HAS ALL THE REQUIRED ATTRIBUTES
    assert_these_attributes(data, name="data object", attr=["X", "Y", "BBOX"])

    # RUN TRAIN STEP
    loss, _, _ = s.run([g.loss, g.train, g.update_moving_avgs],
                       feed_dict={g.X          : data.X,
                                  g.Y          : data.Y,
                                  g.BBOX       : data.BBOX,
                                  g.is_training: True,
                                  g.alpha      : alpha})
    return loss



# ==============================================================================
#                                                         IN_SESSION_PREDICTIONS
# ==============================================================================
def in_session_predictions(s, data):
    """ Given an existing session, it runs predictions on the data.
    
    Args:
        s:          (Tensorflow Session)
        data:       (DataObj) The data to make predictions on. Must contain the
                    attribute X, and batchsize
    Returns:
        (tuple) tuple of two arrays:
            digits, bboxes
    """
    # GET GRAPH OPS BY NAME
    g = GraphOps(s.graph, "is_training", "X", "Y", "BBOX", "digit_preds",
                 "bbox_logits")
    
    # INITIALIZE THE PREDICTIONS ARRAYS
    digit_preds = np.empty(shape=[data.n_samples, 5], dtype=np.int32)
    bbox_preds = np.zeros(shape=[data.n_samples, 24], dtype=np.float32)

    # CALCULATE PREDICTIONS IN BATCHES
    for batch_n in range(data.n_batches):
        batch = data.create_batch(batch_n=batch_n, augment=False)
        d_preds, b_preds = s.run([g.digit_preds, g.bbox_logits],
                                 feed_dict={g.X          : batch.X,
                                            g.Y          : batch.Y,
                                            g.BBOX       : batch.BBOX,
                                            g.is_training: False,
                                            })
        i,j = data.batch_indices(batch_n)
        digit_preds[i:j] = d_preds
        bbox_preds[i:j] = b_preds
 
    # RETURN THE PREDICTIONS
    return digit_preds, bbox_preds


# ==============================================================================
#                                                 MINIMAL_IN_SESSION_PREDICTIONS
# ==============================================================================
def minimal_in_session_predictions(s, X):
    """ Given an existing session, and array of image(s) it runs predictions on
        the data.
        
        This is a minimal version of in_session predictions, where it does not
        expect a DataObj object.
        
        This one is only recomended for when predicting over a small number of
        image samples. If you need to predict over a very large number of
        samples, eg, thousands of samples, then it is recomended you use
        `in_session_predictions()` instead, which makes use of batching.

    Args:
        s:          (Tensorflow Session)
        X:          (array) The images array of shape [n_samples, width, height]

    Returns:
        (tuple) tuple of two arrays: (digits, bboxes)
    """
    # GET GRAPH OPS BY NAME
    g = GraphOps(s.graph, "is_training", "X", "digit_preds", "bbox_logits")
    
    # CALCULATE PREDICTIONS
    digit_preds, bbox_preds = s.run([g.digit_preds, g.bbox_logits],
                             feed_dict={g.X: X, g.is_training: False})

    digit_preds = digit_preds.astype(dtype=np.int32)
    bbox_preds = bbox_preds.astype(dtype=np.float32)
    
    # RETURN THE PREDICTIONS
    return digit_preds, bbox_preds

