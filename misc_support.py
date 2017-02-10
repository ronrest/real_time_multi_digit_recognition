from __future__ import print_function
from scipy import ndimage
from six.moves import cPickle as pickle
from main import range
from main import os
from main import copy


# # ==============================================================================
#                                                               MAYBE LOAD EVALS
# ==============================================================================
def maybe_load_evals(settings, paths):
    """
    Initialize the evals, and max_evals dictionaries.

    - If settings["load_previous"] is True and the evals files exist, then it
      loads the previous evals files.
    - If the above is true, and also settings["load_best"] is true, then it
      loads the previous max evals files as the current evals values.
    - Otherwise it returns blank dictionaries for evals and max_evals
    
    Args:
        settings: settings dictionary
            Should contain the following keys:
            - "load_previous"
            - "load_best"
        paths: dictionary of file paths
            Should contain the following key:
            -"eval_file" Full file path to the eval file
    """
    # TODO: FInd a way to merge this with initialize_eval_metrics()
    exists = os.path.exists
    if settings["load_previous"] and exists(paths["eval_file"]):
        if settings["load_best"]:
            print("Loading best evals")
            evals = pickle2obj(paths["eval_file"] + "_max")
            max_evals = copy.deepcopy(evals)
        else:
            print("Loading last evals")
            evals = pickle2obj(paths["eval_file"])
            max_evals = pickle2obj(paths["eval_file"] + "_max")
    else:
        print("Using new evals")
        evals = {}
        max_evals = {}
    return (evals, max_evals)


# ==============================================================================
#                                                        INITIALIZE_EVAL_METRICS
# ==============================================================================
def initialize_eval_metrics(eval_metrics={}):
    # TODO: FInd a way to integrate this with maybe_load_evals()
    if eval_metrics is None:
        return None
    eval_metrics = copy.deepcopy(eval_metrics)
    # Assign blank lists if the particular metric is not present
    eval_metrics["losses"] = eval_metrics.get("losses", [])
    eval_metrics["valid_accuracies"] = eval_metrics.get("valid_accuracies", [])
    eval_metrics["train_accuracies"] = eval_metrics.get("train_accuracies", [])
    eval_metrics["alpha"] = eval_metrics.get("alpha", [])
    eval_metrics["time"] = eval_metrics.get("time", [])
    return eval_metrics



# ==============================================================================
#                                                         SAVE_DICT_AS_TEXT_FILE
# ==============================================================================
def save_dict_as_text_file(d, f):
    """Takes a dictionary and saves the key=value pairs as lines of text
    in a human readable text file.

    Args:
        d: the dictionary
        f: the path to the file you want to save to. (assumes the parent
           directory has been already created):
    """
    dict_str = ""
    for key in d.keys():
        dict_str += "{} = {}\n".format(str(key), str(d[key]))
    
    with open(f, mode="w") as textFile:
        textFile.write(dict_str)


# ==============================================================================
#                                                                        RESCALE
# ==============================================================================
def rescale(a, new=(-0.5, 0.5), old=None):
    """ Takes a numpy array, and a tuple of the new range of values to scale
    the olds values to. (optionally) it also takes a tuple of the existing
    range of values, otherwise it uses the min and max values in the array
    as the range.
    """
    if old is None:
        old = (a.min(), a.max())
    old_min = old[0]
    new_min = new[0]
    new_range = new[1] - new[0]
    old_range = old[1] - old[0]
    return new_min + (a - old_min)/float(old_range) * new_range



# ==============================================================================
#                                                                  RESCALE_ARRAY
# ==============================================================================
def rescale_vals(val, omin, omax, nmin=0, nmax=1):
    """ Rescales a scalar, or an array of values to be between certain values.

    Args:
        val:   (scalar, or array) value(s) you want to rescale.
        omin:  Minimum value (or the zero point) of original scale
        omax:  Maximum value of original scale
        nmin:  Minimum value (or the zero point) of the new scale
        nmax:  Maximum value of the new scale

    Returns:
        float, or array of floats of rescaled values.
    """
    orange = omax - omin  # original range
    nrange = nmax - nmin  # new range
    return nmin + ((val - omin) / float(orange)) * nrange


