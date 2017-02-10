from __future__ import print_function
from six.moves import cPickle as pickle
import os
from PIL import Image
from main import PRINT_WIDTH
from support import limit_string

# ==============================================================================
#                                                                    MAYBE_MKDIR
# ==============================================================================
def maybe_mkdir(path):
    """ Checks if a directory path exists on the system, if it does not, then
        it creates that directory (and any parent directories needed to
        create that directory)
    """
    if not os.path.exists(path):
        os.makedirs(path)


# ==============================================================================
#                                                           GET_PARENT_DIRECTORY
# ==============================================================================
def get_parent_directory(file):
    """ Given a file path, it returns the parent directory of that file. """
    return os.path.dirname(file)


# ==============================================================================
#                                                              MAYBE_MAKE_PARDIR
# ==============================================================================
def maybe_make_pardir(file):
    """ Takes a path to a file, and creates the necessary directory structure
        on the system to ensure that the parent directory exists (if it does
        not already exist)
    """
    maybe_mkdir(get_parent_directory(file))


# ==============================================================================
#                                                                     OBJ2PICKLE
# ==============================================================================
def obj2pickle(obj, file, protocol=-1, verbose=False):
    """Saves an object as a binary pickle file to the desired file path.

    Args:
        obj:        The python object you want to save.
        file:       (string)
                    File path of file you want to save as.  eg /tmp/myFile.pkl
        protocol:   (int)(default=-1)
                    0  = original ASCII protocol and is backwards compatible
                         with earlier versions of Python.
                    1  = old binary format which is also compatible with earlier
                         versions of Python.
                    2  = provides much more efficient pickling of new-style
                         classes. compatible with Python >= 2.3
                    -1 = uses the highest (latest) protocol available. This is
                         the default value for this function.
    """
    # ==========================================================================
    if verbose:
        # TODO: handle case where PRINT_WIDTH is less than 21
        max_len = PRINT_WIDTH - 21
        s = file if len(file) <= max_len else ("..." + file[-max_len:])
        print("Saving: " + s, end="")
    
    # maybe make the parent dir
    pardir = os.path.dirname(file)
    if not (pardir == ""):
        maybe_mkdir(pardir)
    
    with open(file, mode="wb") as fileObj:
        pickle.dump(obj, fileObj, protocol=protocol)
    
    if verbose:
        print(" -- [DONE]")


# ==============================================================================
#                                                                     PICKLE2OBJ
# ==============================================================================
def pickle2obj(file, verbose=False):
    """Takes a filepath to a pickle object, and returns a python object
    specified by that pickle file. """
    if verbose:
        # TODO: handle case where PRINT_WIDTH is less than 22
        max_len = PRINT_WIDTH - 22
        s = file if len(file) <= max_len else ("..." + file[-max_len:])
        print("Loading: " + s, end="")
        # s = file if len(file) < 41 else (file[:10] + "..." + file[-28:])
        # print("Loading ", s, end="")
    
    with open(file, mode="rb") as fileObj:
        obj = pickle.load(fileObj)
    
    if verbose:
        print(" -- [DONE]")
    
    return obj


# ===============================================================================
#                                                           SHOW_IMAGE_FROM_FILE
# ===============================================================================
def show_image_from_file(f):
    img = Image.open(f)
    img.show()


# ===============================================================================
#                                                                IMAGE_FROM_FILE
# ===============================================================================
def image_from_file(f):
    return Image.open(f)


# ==============================================================================
#                                                         SAVE_DICT_AS_TEXT_FILE
# ==============================================================================
def save_dict_as_text_file(d, f):
    """ Takes a dictionary and saves the key=value pairs as lines of text
        in a human readable text file.

    Args:
        d: (dict) the dictionary
        f: (str) the path to the file you want to save to. Creates the necessary
           parent directories in order to save the file.
    """
    # CREATE PARENT DIRECTORIES -if needed
    pardir = get_parent_directory(f)
    maybe_mkdir(pardir)
    
    # SAVE AS TEXT FILE
    with open(f, mode="w") as textFile:
        for key in sorted(d.keys()):
            textFile.write("{}   {}\n".format(str(key), str(d[key])))
            

