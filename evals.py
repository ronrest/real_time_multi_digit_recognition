from __future__ import print_function
from main import np
from main import os
import copy

from nnet.graphops import GraphOps
from file_support import pickle2obj, obj2pickle
from support import verbose_print, verbose_print_done, limit_string
from main import PRINT_WIDTH


# ==============================================================================
#                                                           PER_ELEMENT_ACCURACY
# ==============================================================================
def per_element_accuracy(a, b, axis=None):
    """ Evaluates The accuracy(ies) of elements of predicted values against the
        ground truth.
        
        Select axis=0 to calculate separate  accuracy for each separate
        column of predictions.
         
    Args:
        a:      (numpy array) truth or predictionc
        b:      (numpy array) predictions or truth
        axis:   (bool) if None (default) it calculates the accuracy over all
                elements. axis=0 calculates separate accuracy for each column.
    Returns:
        if axis=None, then returns a scalar.
        if axis=0, then returns a numpy array with same number of elements
        as columns in `a`.
    """
    return (a == b).mean(axis=axis)


# ==============================================================================
#                                                              FULL_ROW_ACCURACY
# ==============================================================================
def full_row_accuracy(a, b):
    """ for each row in a and b, it is only considered to be correct if
        every single element in that row matches up. If there is even one
        mismatch in the row, then the whole row is considrered False.
        
        Returns the proportion of rows that match up completely.
        
    Args:
        a:      (numpy array) truth or predictionc
        b:      (numpy array) predictions or truth
        .
    Returns:
        Float
    """
    return (a == b).all(axis=1).mean()


# ==============================================================================
#                                                                      BATCH_IOU
# ==============================================================================
def batch_iou(a, b, epsilon=1e-8):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)
    
    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    
    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0
    
    area_overlap = width * height
    
    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap
    
    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


# ==============================================================================
#                                                         BATCH_MULTI_COLUMN_IOU
# ==============================================================================
def batch_multi_column_iou(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a several bounding
        boxes in groups of 4 columns. Where each group of 4 columns represents
        a single bounding box as:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each bounding box in the
        data.
        
        The shape of output is [n_samples, n_bboxes]

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each bounding box.
    """
    n_bboxes = a.shape[1] // 4
    n_samples = a.shape[0]
    ious = np.empty(shape=[n_samples, n_bboxes])
        
    for i in range(n_bboxes):
        ious[:,i] = batch_iou(a[:, 4*i: 4+4*i], b[:, 4*i: 4+4*i])
    
    return ious


# ==============================================================================
#                                                                        AVG_IOU
# ==============================================================================
def avg_iou(preds, Y, axis=None):
    return batch_iou(preds, Y).mean(axis=axis)


# ==============================================================================
#                                                           AVG_MULTI_COLUMN_IOU
# ==============================================================================
def avg_multi_column_iou(preds, Y, axis=None):
    return batch_multi_column_iou(preds, Y).mean(axis=axis)


# ==============================================================================
#                                                                  STR_ELSE_NONE
# ==============================================================================
def str_else_none(a, f, e):
    """ If `a` is not None, then it returns `a` formatted
        as a string, specified by the formatting rule
        specified by `f`, otherwise it returns the
        alternative string specified by `e`
    """
    return f.format(a) if a is not None else e


# ##############################################################################
#                                                                          EVALS
# ##############################################################################
class Evals(object):
    def __init__(self, d=None, pickle=None, verbose=False):
        """ Creates an Evals object to store evaluation metrics for each epoch.
        
        Args:
            d:          (dict or None)(optional) - initialize Evals object from
                        a dictionary
            pickle:     (str or None) (optional) path to a pickle file of a
                        dictionary to initialize the Evals object.
            verbose:    (bool)
        """
        self.stuff = dict()
        self._initialized = True
        
        # INITIAL BLANK VALUES
        self.pda_train = []
        self.pda_valid = []
        self.wna_train = []
        self.wna_valid = []
        self.iou_train = []
        self.iou_valid = []
        self.time_pred = []
        self.time_train = []
        self.loss = []
        self.alpha = []
        
        # LOAD EVALS FROM DICTIONARY
        if d is not None:
            verbose_print("Loading evals from a dictionary", verbose=verbose, end="")
            self.stuff.update(copy.deepcopy(d))

        # LOAD EVALS FROM PICKLE FILE (of a dictionary)
        elif pickle is not None:
            short_path = limit_string(pickle, tail=PRINT_WIDTH-32)
            verbose_print("Loading evals from " + short_path, verbose, end="")
            if os.path.exists(pickle):
                d = pickle2obj(pickle)
                self.stuff.update(copy.deepcopy(d))
            else:
                verbose_print("\n-- file does not exist. Creating blank Evals", verbose, end="")
        else:
            verbose_print("Creating blank Evals", verbose, end="")

        verbose_print_done(verbose)

    def __getattr__(self, key):
        """ Get items using dot notation """
        return self.stuff[key]
    
    def __setattr__(self, key, value):
        """ Set attributes (that will be stored in the dict `stuff` using dot
            notation
        """
        if self.__dict__.has_key('_initialized'):
            self.stuff[key] = value
        else:
            # allows attributes to be set in the __init__ method
            return dict.__setattr__(self, key, value)
    
    def __getitem__(self, key):
        """ Get items using dictionary notation """
        return self.stuff[key]
    
    def __setitem__(self, key, val):
        """ Set items using dictionary notation """
        self.stuff[key] = val
    
    def make_copy(self):
        """Returns a deep copy of this object"""
        return Evals(d=self.stuff)
    
    def save_dict_pickle(self, f, verbose=False):
        short_path = limit_string(f, front=10, tail=31)
        verbose_print("Saving Evals to " + short_path, verbose, end="")
        obj2pickle(self.stuff, file=f)
        verbose_print_done(verbose)
    
    def as_dict(self):
        return self.stuff
    
    def newest_is_max(self):
        """ Returns true, if the latest entry has the highest Whole Number
            Accuracy when evaluated on the validation dataset.
        """
        return max(self.wna_valid) == self.wna_valid[-1]
        
    @property
    def epochs(self):
        return len(self.loss)
    
    def append(self, **kwargs):
        """ Given a set of keyword, value pairs, it appends the value to the
            end of the list that is in the attribute specified by the keyword.
            
            If the keyword is not an already existing attribute of the Evals
            object, then it creates one, and initializes a new list, with the
            fisrt item being the value provided.
        
        Args:
            **kwargs:
        """
        # GIVE FEEDBACK ABOUT MISSING OR EXRTANEOUS KEYS
        inner_keys = set(self.stuff.keys())
        new_keys = set(kwargs.keys())
        missing = inner_keys.difference(new_keys)
        extraneous = new_keys.difference(inner_keys)
        overlapping = inner_keys.intersection(new_keys)
        if missing != set():
            print("WARNING!: Missing the following keys: ", sorted(list(missing)))
        if extraneous != set():
            print("WARNING!: The following keys are extraneous: ", sorted(list(extraneous)))
        
        # APPEND VALUES TO RELEVANT LIST
        for key in overlapping:
            self[key].append(kwargs[key])
        
        # INITIALIZE EXTRANEOUS KEY VALUES IN A NEW LIST
        for key in extraneous:
            self[key] = [(kwargs[key])]
    
    def print_header(self):
        """ Prints out a header string that details the column names that will
            be used in `print_line()`
        """
        h  = "-------------------------+--------------+----------------+----------------+\n"
        h += "               TIME (ms) |       IoU    |      PDA       |       WNA      |\n"
        h += " ep  LOSS    train  pred | train  valid | train    valid | train    valid |\n"
        h += "-------------------------+--------------+----------------+----------------+"
        print(h)

    def print_loss(self, loss,end="\n"):
        line = "     {:1.4f}              |              |                |                |"
        print(line.format(loss), end=end)

    def print_line(self, end="\n"):
        """ Prints a line of the evaluation metrics. from the latest epoch.
            (To be used in conjunction with print_header() to print out the
            labels of each column)
        """
        s = ""
        s += str_else_none(self.epochs,         f="{:3.0f} ",   e="    ")
        s += str_else_none(self.loss[-1],       f=" {:1.4f} ",  e="        ")
        s += str_else_none(self.time_train[-1], f=" {:2.2f} ",  e="       ")
        s += str_else_none(self.time_pred[-1],  f=" {:1.2f} |", e="      |")
        s += str_else_none(self.iou_train[-1],  f=" {:1.3f} ",  e="        ")
        s += str_else_none(self.iou_valid[-1],  f=" {:1.3f} |", e="        |")
        s += str_else_none(self.pda_train[-1],  f=" {:2.3f} ",  e="        ")
        s += str_else_none(self.pda_valid[-1],  f=" {:2.3f} |", e="        |")
        s += str_else_none(self.wna_train[-1],  f=" {:2.3f} ",  e="        ")
        s += str_else_none(self.wna_valid[-1],  f=" {:2.3f} |", e="        |")
    
        print(s, end=end)
    
    
# ==============================================================================
#                                                               COMPARISONS_FILE
# ==============================================================================
def comparisons_file(opts, paths):
    """ Saves a text file
            accuracy__iou__modelname.txt
        Inside the file, is the eval metrics for the best epoch.
    """
    evalsmax = Evals(pickle=paths.evals_max)
    
    filename = "{acc}__{iou}__{name}.txt".format(
        acc=str(round(evalsmax.wna_valid[-1], 3)).zfill(7),
        iou=str(round(evalsmax.iou_valid[-1], 3)).zfill(5),
        name=opts.output_name,
        )
    path = os.path.join(paths.model_comparisons_dir, filename)
    with open(path, mode="w") as textFile:
        for key in sorted(evalsmax.as_dict().keys()):
            textFile.write("{}   {}\n".format(key, evalsmax[key][-1:]))

