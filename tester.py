"""
################################################################################
                            EVALUATE ON TEST DATASET
################################################################################
Saves several images:
- Grid of predictions on test set
- Grid of worst performing bounding box predictions
- Grid of incorrect digit predictions

Prints out the following metrics:

- PDA
- WNA
- IoU

See the README file for how to use this script.
################################################################################
"""
from __future__ import print_function, division

from main import tf
from main import np
from main import os

from nnet.sessions import in_session_predictions

# Visualisation Imports
import matplotlib as mpl

mpl.use('Agg')  # Matplotlib in non-interactive mode

# Neural Net Imports
from nnet.graphops import GraphOps
from nnet.sessions import tf_initialize_vars_from_file
from graphs import create_graph, model_a, model_b, model_c, model_d

# Misc Imports
from main import PRINT_WIDTH
from support import print_headers
from support import verbose_print_done, verbose_print

from support.dataObjects import DataObj
from file_support import pickle2obj, obj2pickle, save_dict_as_text_file
from evals import avg_multi_column_iou, per_element_accuracy, full_row_accuracy

from vis import grid_of_sample_images
from support import array_of_digit_arrays_to_str, digit_array_to_str
from vis import grid_of_sample_bboxes, GREEN, RED, NICE_BLUE


class Settings(object):
    pass

# ==============================================================================
#                                                                       SETTINGS
# ==============================================================================
verbose = True
limit = None        # place limit on the amount of data (for DEBUGGING purposes)

# ESTABLISH MODEL SETTINGS
settings = Settings()
settings.conv_dropout = 0.1
settings.fc_dropout = 0.1
settings.image_chanels = 1
settings.image_size = (54, 54)

# ==============================================================================
#                                                                           DATA
# ==============================================================================
data_dir = "data"
X_file = os.path.join(data_dir, "X_test_cropped64.pickle")
Y_file = os.path.join(data_dir, "Y_test.pickle")

verbose_print("Loading data", verbose=verbose, end="")
data = DataObj(X=pickle2obj(X_file),
               Y=pickle2obj(Y_file),
               batchsize=128)
verbose_print_done(verbose)

data.limit_samples(n=limit, verbose=verbose)

verbose_print("Performing center crops", verbose=verbose, end="")
data.do_center_crops()
verbose_print_done(verbose)

# ==============================================================================
#                                                              GRAPH AND SESSION
# ==============================================================================
model = model_a
checkpoint_dir = "results/A_02/checkpoints/"
checkpoint_file = os.path.join(checkpoint_dir, "checkpoint_max.chk")

# CREATE TENSORFLOW GRAPH
graph = create_graph(logit_func=model, settings=settings)

# PREPARE SESSION
print_headers("TENSORFLOW SESSION", border="=", width=PRINT_WIDTH)

with tf.Session(graph=graph) as sess:
    # GET IMPORTANT OPERATIONS AND TENSORS FROM GRAPH
    g = GraphOps(graph, "X", "Y", "BBOX", "is_training", "digit_logits",
                 "bbox_logits")
    
    # INITIALIZE VARIABLES
    saver = tf.train.Saver(name="saver")
    tf_initialize_vars_from_file(f=checkpoint_file, s=sess, saver=saver,
                                 verbose=verbose)

    # PREDICT
    verbose_print("Making predictions", verbose=verbose, end="")
    digits, bboxes = in_session_predictions(s=sess, data=data)
    verbose_print_done(verbose)

    # EVALUATION METRICS
    verbose_print("Calculating WNA Accuracies", verbose=verbose, end="")
    wna = full_row_accuracy(a=digits, b=data.Y)
    verbose_print_done(verbose)

    verbose_print("Calculating PDA Accuracies", verbose=verbose, end="")
    pda = per_element_accuracy(a=digits, b=data.Y, axis=0)
    verbose_print_done(verbose)

    verbose_print("Calculating IoU Scores", verbose=verbose, end="")
    iou = avg_multi_column_iou(preds=bboxes[:,4:], Y=data.digit_bboxes, axis=1)
    verbose_print_done(verbose)

    # PRINT EVALUATION METRICS
    print_headers("EVALUATION ON TEST DATA", align="center")
    print("WNA: ", 100*wna)
    print("PDA: ", 100*pda.mean())
    print("IOU: ", iou.mean())


# CONVERT LABELS TO SOMETHING MORE USEFUL
pred_digits = array_of_digit_arrays_to_str(digits, null=10)
pred_bboxes = bboxes[:, 4:]



# ==============================================================================
#                                             VISUALISE PREDICTIONS ON TEST DATA
# ==============================================================================
n_samples = 25
labels = np.round(iou[:n_samples], 3)
labels2a = pred_digits[:n_samples]
labels2b = array_of_digit_arrays_to_str(data.Y[:n_samples], null=10)
labels2 = ["{} ({})".format(labels2a[i], labels2b[i]) for i in range(n_samples)]

grid_of_sample_bboxes(data.X[:n_samples],
                      bboxes=data.digit_bboxes[:n_samples],
                      bboxes2=pred_bboxes[:n_samples],
                      fill=None, outline=GREEN,
                      fill2=None, outline2=RED,
                      gridsize=(5, 5),
                      proportional=True,
                      labels=labels,
                      labels2=labels2,
                      label_color="#000000",
                      label2_color=NICE_BLUE,
                      label_font_size=7,
                      title="Sample of Predictions from Test Data",
                      saveto="imgs/test_predictions.png")


# ==============================================================================
#                                             VISUALISE DIFFICULT BOUNDING BOXES
# ==============================================================================
n_samples = 16

# DIFICULT SAMPLES
i_dificult = np.argwhere(iou < 0.55).flatten()
dif_data = data.extract_items(i_dificult, deepcopy=True, verbose=verbose)
dif_pred_bboxes = pred_bboxes[i_dificult]

# LABELS
labels = np.round(iou[i_dificult], 3)
labels2a = pred_digits[i_dificult]
labels2b = array_of_digit_arrays_to_str(dif_data.Y, null=10)
labels2 = ["{} ({})".format(labels2a[i], labels2b[i]) for i in range(n_samples)]

# DRAW GRID OF DIFICULT BOUNDING BOXES ON TEST DATA
grid_of_sample_bboxes(dif_data.X,
                      bboxes=dif_data.digit_bboxes,
                      bboxes2=dif_pred_bboxes,
                      fill=None, outline=GREEN,
                      fill2=None, outline2=RED,
                      gridsize=(4, 4),
                      proportional=True,
                      labels=labels,
                      labels2=labels2,
                      label_color="#000000",
                      label2_color=NICE_BLUE,
                      label_font_size=7,
                      title="Sample of Bounding Boxes with IoU < 0.55",
                      saveto="imgs/dificult_test_bboxes.png",
                      )


# ==============================================================================
#                                           VISUALISE DIFICULT DIGIT PREDICTIONS
# ==============================================================================
n_samples = 25

# DIFICULT SAMPLES
i_dificult = np.argwhere((digits != data.Y).any(axis=1)).flatten()
dif_data = data.extract_items(i_dificult, deepcopy=True, verbose=verbose)
dif_pred_digits = pred_digits[i_dificult]

# LABELS
labels_a = pred_digits[i_dificult]
labels_b = array_of_digit_arrays_to_str(dif_data.Y, null=10)
labels = ["{} ({})".format(labels_a[i], labels_b[i]) for i in range(dif_data.n_samples)]


# GRID OF DIFICULT SAMPLES
grid_of_sample_images(dif_data.X,
                      labels=labels,
                      label_font_size=9,
                      label_color=NICE_BLUE,
                      gridsize=(5,5),
                      title="Sample of Incorrectly Predicted Numbers on Test Set",
                      saveto="imgs/dificult_test_digits.png", random=False)


# # ==============================================================================
# #                                           TRAINING CURVES
# # ==============================================================================
# from vis import plot_training_curves
# from evals import Evals
#
# model_name = "A_02"
# evals_dir = "results/{}/evals/".format(model_name)
# evals_file = os.path.join(evals_dir, "evals.pickle")
#
# evals = Evals(pickle=evals_file, verbose=True)
#
# plot_training_curves(evals, crop=(None, None), saveto=os.path.join(evals_dir, "learning_curves.png"))

