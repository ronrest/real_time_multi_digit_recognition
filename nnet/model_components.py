from __future__ import print_function, absolute_import
from main import tf
from main import np
from main import range
from .initializers import *


# ==============================================================================
#                                                                 CALCULATE_LOSS
# ==============================================================================
def multi_digit_loss(logits_list, Y, max_digits=5, name="loss"):
    """ Calculates the loss for the multi-digit recognition task,
        given a list of the logits for each digit, and the correct
        labels.

    Args:
        logits:         (list of tensors) list of the logits from each of the
                        branches
        Y:              (tensor) correct labels, shaped as [n_batch, max_digits]
        name:           (str) Name for the scope of this loss.

    Returns:
        (tensor) the loss
    """
    with tf.name_scope(name) as scope:
        # LOSSES FOR EACH DIGIT BRANCH
        losses = [None] * (max_digits)
        losses[0] = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits_list[0], Y[:, 0])
        losses[1] = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits_list[1], Y[:, 1])
        losses[2] = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits_list[2], Y[:, 2])
        losses[3] = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits_list[3], Y[:, 3])
        losses[4] = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits_list[4], Y[:, 4])
        # AVERAGE LOSS
        loss = sum(losses) / float(max_digits)
        loss = tf.reduce_mean(loss, name=scope)
    return loss


