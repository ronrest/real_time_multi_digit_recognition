from __future__ import print_function, division

################################################################################
#                                                                        IMPORTS
################################################################################
from main import tf

# Visualisation Imports
import matplotlib as mpl
mpl.use('Agg') # Matplotlib in non-interactive mode
from vis import epoch_visualisations, plot_training_curves

# Neural Net Imports
from nnet.graphops import GraphOps
from nnet.sessions import train_step, tf_initialize_vars_from_file
from graphs import create_graph, model_a, model_b, model_c, model_d

# Data Imports
from support.dataObjects import DataObjects
from file_support import pickle2obj, obj2pickle, save_dict_as_text_file

# Evaluation Imports
from nnet.sessions import in_session_predictions
from evals import per_element_accuracy, full_row_accuracy
from evals import avg_multi_column_iou
from evals import Evals, comparisons_file

# Misc Imports
from main import PRINT_WIDTH
from support import print_headers
from support import Timer
from support import verbose_print, verbose_print_done
from settings import parse_settings, establish_paths
from support import assert_these_attributes


# ==============================================================================
#                                                              EPOCH_EVALUATIONS
# ==============================================================================
def epoch_evaluations(s, evals, loss, alpha, tr_time, data, paths):
    """ Perform evaluation metrics on the validation data, and the subset of the
        training data to be used for evaluation.
        
        It places the results in the `evals` object, and creates visualisations
        of predictions vs reality for bounding boxes and digit recognition.
        
        It saves a snapshot of the evals file.
        
        And also creates a plot of the training curves.
    
    Args:
        s:          (tensorflow session) The current open session.
        evals:      (evals object)
        loss:       (float) The last loss value in training.
        alpha:      (float) The current learning rate
        tr_time:    (float) in milliseconds, how long did it take to train a
                    single sample (on average) for the last last batch of data.
        data:       (DataObjects object) should contain the following attributes
                    - valid
                    - train_eval
        paths:      (Paths object) should contain the following attributes with
                    filepaths as values:
                    - learning_curves
                    - evals
                    - evals_max
    """
    # EVALS ON TRAIN DATA
    digit_preds_tr, bbox_preds_tr = in_session_predictions(s=s, data=data.train_eval)
    pda_tr = 100 * per_element_accuracy(digit_preds_tr, data.train_eval.Y)
    wna_tr = 100 * full_row_accuracy(digit_preds_tr, data.train_eval.Y)
    ious_tr = avg_multi_column_iou(bbox_preds_tr, data.train_eval.BBOX, axis=0)
    w_iou_tr = ious_tr[0]
    digit_iou_tr = ious_tr[1:].mean()

    # EVALS ON VALID DATA
    timer = Timer()
    timer.start()
    digit_preds_va, bbox_preds_va = in_session_predictions(s=s, data=data.valid)
    avg_pred_time = 1000 * timer.stop() / float(data.valid.n_samples)
    
    pda_va = 100 * per_element_accuracy(digit_preds_va, data.valid.Y)
    wna_va = 100 * full_row_accuracy(digit_preds_va, data.valid.Y)
    ious_va = avg_multi_column_iou(bbox_preds_va, data.valid.BBOX, axis=0)
    w_iou_va = ious_va[0]
    digit_iou_va = ious_va[1:].mean()
    
    # ADD NEW SET OF EVALUATION ENTRIES TO EVALS OBJECT
    evals.append(loss=loss, alpha=alpha,
                 pda_train=pda_tr, pda_valid=pda_va,
                 wna_train=wna_tr, wna_valid=wna_va,
                 iou_train=digit_iou_tr, iou_valid=digit_iou_va,
                 time_train=tr_time, time_pred = avg_pred_time,
                 )
    
    # SAVE SNAPSHOT OF EVALS
    evals.save_dict_pickle(f=paths.evals)
    if evals.newest_is_max():
        evals.save_dict_pickle(f=paths.evals_max)
        printout_end = " *\n"   # Indicate the current max in the evals printout
    else:
        printout_end = "\n"

    # PRINTOUTS AND VISUALISATIONS
    evals.print_line(end=printout_end)
    epoch_visualisations(path=paths.epoch_vis,
                         epoch=evals.epochs,
                         data=data.valid,
                         bboxes_pred=bbox_preds_va,
                         digits_pred=digit_preds_va)
    plot_training_curves(evals, crop=(None, None), saveto=paths.learning_curves)
    

################################################################################
#                                                                    RUN_SESSION
################################################################################
def run_session(graph, data, paths, alpha=0.001, epochs=1):
    """ Runs a training session.
    
    Args:
        graph: (Tensorflow graph) The graph that contains the model
        data:  (DataObjects) DataObjects object with the following attributes:
               - train (DataObj) containing the training data
               - valid (DataObj) containing the validation data
               - train_eval (DataObj) containing the portion of train data to be
                 used for evaluation.
        paths: (Paths object) The paths, containing the following attributes:
                - checkpoint
                - checkpoint_max
                - evals
                - evals_max
                - learning_curves
                - epoch_vis
                
        alpha:  (float) learning rate
        epochs: (int) number of epochs to run.
    """
    assert_these_attributes(paths, "Paths Object",
                            ["checkpoint", "checkpoint_max",
                             "evals", "evals_max",
                             "learning_curves", "epoch_vis"])
    assert_these_attributes(data, "DataObjects object",
                            ["train", "valid", "train_eval"])

    # PREPARE SESSION
    timer = Timer()
    print_headers("SESSION", border="=", width=PRINT_WIDTH)
    print("Training on {} samples in batches of {}".format(data.train.n_samples,
                                                           data.train.batchsize))
    print("Alpha: ", alpha)
    evals = Evals(pickle=paths.evals, verbose=True)
    
    with tf.Session(graph=graph) as sess:
        # GET IMPORTANT OPERATIONS AND TENSORS FROM GRAPH
        g = GraphOps(graph, "X", "Y", "BBOX", "is_training", "alpha",
                     "train", "loss", "digit_logits", "bbox_logits")
        g.update_moving_avgs = tf.group(*tf.get_collection("update_moving_averages"))
        
        # INITIALIZE VARIABLES
        saver = tf.train.Saver(name="saver")
        tf_initialize_vars_from_file(f=paths.checkpoint, s=sess, saver=saver,
                                     verbose=verbose)
        
        # TRAIN FOR SEVERAL EPOCHS
        evals.print_header()
        for epoch in range(epochs):
            # PREPARE FOR A NEW EPOCH
            timer.start()
            data.train.shuffle()
            
            # TRAIN IN BATCHES
            for batch_n in xrange(data.train.n_batches):
                batch = data.train.create_batch(batch_n=batch_n, augment=True)
                loss = train_step(s=sess, g=g, data=batch, alpha=alpha)
                
                # INTERMITTENT FEEDBACK ON PROGRESS
                n_feedback_steps = 4
                feedback_steps = int(data.train.n_batches / n_feedback_steps)
                if batch_n % feedback_steps == 0:
                    evals.print_loss(loss=loss)
            
            # EVALUATIONS AT END OF EACH EPOCH
            avg_train_time = 1000 * timer.stop() / float(data.train.n_samples)
            epoch_evaluations(s=sess, evals=evals, loss=loss, alpha=alpha,
                              tr_time=avg_train_time,  data=data, paths=paths)
            
            # SAVE CHECKPOINTS
            saver.save(sess, paths.checkpoint)
            if evals.newest_is_max():
                saver.save(sess, paths.checkpoint_max)


################################################################################
#                                                                           MAIN
################################################################################
if __name__ == "__main__":
    # SETTINGS
    verbose= True
    opts = parse_settings()
    print_headers(opts.output_name, border="#", align="center", width=PRINT_WIDTH)
    verbose_print("Establishing paths", verbose, end="")
    paths = establish_paths(output_name=opts.output_name, input=opts.input_dir)
    verbose_print_done(verbose)

    # TRAIN DATA
    data = DataObjects()
    data.set_train_data(X=pickle2obj(paths.X_train, verbose=True),
                     Y=pickle2obj(paths.Y_train, verbose=True),
                     batchsize=opts.batch_size)
            
    # VALID DATA
    data.set_valid_data(n=opts.valid_size, random=False, batchsize=128, verbose=verbose)

    # LIMIT TRAIN DATA - eg during development and debugging
    limit = opts.data_size
    data.train.limit_samples(n=limit, verbose=verbose)

    # PORTION OF THE TRAINING DATA USED FOR EVALUATION
    data.set_train_eval_data(n=1024, random=False, random_transforms=True,
                             batchsize=128, verbose=verbose)

    # CREATE TENSORFLOW GRAPH
    print("USING MODEL: ", opts.model)
    models = {"a": model_a,
              "b": model_b,
              "c": model_c,
              "d": model_d}
    graph = create_graph(logit_func=models[opts.model], settings=opts)

    # RUN TENSORFLOW TRAINING SESSION
    run_session(graph=graph, data=data, paths=paths, alpha=opts.alpha, epochs=opts.epochs)
    
    # SAVE SETTINGS TO TEXT AND PICKLE FILES
    save_dict_as_text_file(d=vars(opts), f=paths.settings_text_file)
    obj2pickle(opts, paths.settings_pickle_file, verbose=verbose)
    
    # CREATE A COMPARISONS FILE
    comparisons_file(opts, paths)
 
