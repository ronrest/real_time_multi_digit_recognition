from __future__ import print_function, absolute_import
import numpy as  np
import os
import h5py
import copy
from PIL import Image

from support import merge_arrays, repeat_array
from main import NULL_BBOX


#  =============================================================================
#                                                              HD5_ELEMENT VALUE
#  =============================================================================
def hd5_element_value(h, path, i):
    """Takes a HD5 object `h`, the internal `path` to the column of data, and
    the index `i` of the element, and returns the value"""
    ref = h[path][i][0]  # HD5 reference to the item
    return h[ref].value  # the actual value of the item


#  =============================================================================
#                                                                       HD5_ITEM
#  =============================================================================
def hd5_item(h, path, i):
    """Takes a HD5 object `h`, the internal `path` to the item of data, and
    the index `i`, and returns the item"""
    ref = h[path][i][0]  # HD5 reference to the item
    return h[ref]  # the group


#  =============================================================================
#                                                          ASCII_ARRAY_TO_STRING
#  =============================================================================
def ascii_array_to_string(a):
    """Takes an array of ascii characters, and converts it to a python string"""
    return ''.join(chr(ch) for ch in a)


#  =============================================================================
#                                                                   PADDED_ARRAY
#  =============================================================================
def padded_array(a, l, pad):
    """ Takes an array `a`, the max length of the array `l`, and the
        value to `pad` with.  Returns an array of length `l`, that adds
        padding to the left (if necessary)

        Example:
        >>> padded_array([1,2,4], l=5, pad=10)
        array([10, 10,  1,  2,  4])
    """
    return np.pad(a[-l:], ((l - len(a), 0)), 'constant', constant_values=(pad))


# ==============================================================================
#                                                     CREATE_SINGLE_BBOXES_ARRAY
# ==============================================================================
def create_single_bboxes_array(labels, bboxes):
    """ takes multiple bounding box arrays, each of shape:
            [n_samples, 4]
        Where for each sample we have a bounding box specified as:
            [x1, y1, x2, y2]
        And merges them all together so we have an array of shape:
            [n_samples, 4 * n_bboxes]

    Args:
        labels: (dict) Dictionary of arrays, containing all the bounding boxes
                named in the `bboxes` argument.
        bboxes: (list of strings) The names of the keys in `labels` that hold
                the bounding box data you want to merge.

    Returns:
        (numpy array) [n_samples, 4 * n_bboxes]
    """
    # The actual arrays containing the bbox data
    bboxes_arrays = [labels[key] for key in bboxes]
    
    # MERGE TOGETHER - all bboxes into a single array
    all_bboxes = merge_arrays(bboxes_arrays, axis=1)
    
    return all_bboxes


################################################################################
#                                                     MAT_DATA_TO_DICT_OR_ARRAYS
################################################################################
def mat_data_to_dict_of_arrays(data_file, limit=None):
    """ Takes the data from the mat file, and converts it to a dictionary of
        numpy arrays that can be used more easily by python.

        Optionally set a `limit` to the number of samples to proccess.
        (eg, for testing on small number of data)
    """
    mat_data = h5py.File(data_file)  # Load the hd5 data
    
    if limit is None:
        num_samples = len(mat_data["digitStruct/name"])
    else:
        num_samples = limit
    
    max_digits = 6  # Max number of digits to represent in digit arrays
    null_digit = 10  # value representing No Digit for this digit position
    null_coord = - 1  # value representing No Coordinate for this digit position
    
    # Initialize dictionary of data
    processed_data = {"file"  : np.empty(shape=num_samples, dtype=np.object),
                      "label" : np.empty(shape=[num_samples, 6],
                                         dtype=np.int32),
                      "N"     : np.empty(shape=num_samples, dtype=np.int32),
                      "left"  : np.empty(shape=[num_samples, 6],
                                         dtype=np.int32),
                      "top"   : np.empty(shape=[num_samples, 6],
                                         dtype=np.int32),
                      "width" : np.empty(shape=[num_samples, 6],
                                         dtype=np.int32),
                      "height": np.empty(shape=[num_samples, 6],
                                         dtype=np.int32)}
    
    for i in range(num_samples):
        # ----------------------------------------------------------------------
        #                                                       Process Filename
        # ----------------------------------------------------------------------
        path = "digitStruct/name"  # Path to data in hd5 file
        ascii = hd5_element_value(mat_data, path=path,
                                  i=i)  # List of ascii chars
        s = ascii_array_to_string(ascii)  # Convert to a string
        processed_data["file"][i] = s  # Add file name to our list
        
        # ----------------------------------------------------------------------
        #                                                     Process Digit Data
        # ----------------------------------------------------------------------
        path = "digitStruct/bbox"
        bbox_data = hd5_item(mat_data, path=path, i=i)
        digit_data_components = ['label', 'top', 'left', 'width', 'height']
        
        for component in digit_data_components:
            cmp_data = bbox_data[component]  # Single component data
            
            # If the length is 1, then the value can be extracted directly.
            # Otherwise it contains an array of references to individual digits.
            if len(cmp_data) < 2:
                vals = cmp_data[0]
            else:
                vals = [mat_data[cmp_data.value[j].item()].value[0][0] for j in
                        range(len(cmp_data))]
            
            # CLEAN UP DIGIT DATA
            # Label data needs to be cleaned up in a different way to the
            # Coordinate data components
            if component == "label":
                # Number of digits
                N = len(vals)
                
                # Make digits representing 0, actually 0
                vals = np.array(vals)
                vals[vals == 10] = 0
                
                # Pad the array to make all samples a consistent array size
                vals = padded_array(vals, l=max_digits, pad=null_digit).astype(
                    np.int32)
            else:
                # Pad the array to make all samples a consistent array size
                vals = padded_array(vals, l=max_digits, pad=null_coord).astype(
                    np.int32)
            
            processed_data[component][i] = vals
        processed_data["N"][i] = N
    
    # Rename "label" key to the more meaningful "digits" key
    processed_data["digits"] = processed_data.pop("label")
    
    return processed_data


# ==============================================================================
#                                                        GET_FULL_BOUNDING_BOXES
# ==============================================================================
def get_full_bounding_boxes(labels):
    """
    Given the dictionary of arrays extracted from the mat file (plus the
    additional key "N", containing the number of digits), it looks at the
    individual digit bounding boxes and calculates the bounding box for the
    entire number.

    Args:
        labels: The dictionary of arrays containing labels

    Returns:
        Numpy array of shape [num_samples, 4]
        Where the the bounding box elements for each training example is
        ordered like this:

            [x1, y1, x2, y2]
    """
    num_samples = labels["N"].shape[0]
    
    # Initialize the array that will store the bounding box data.
    bboxes = np.zeros(shape=[num_samples, 4], dtype=np.int32)
    
    for i in range(num_samples):
        N = int(labels["N"][i])
        
        # Get the individual digit bounding boxes data
        left = labels["left"][i][-N:]
        top = labels["top"][i][-N:]
        width = labels["width"][i][-N:]
        height = labels["height"][i][-N:]
        
        # calculate the bounding box data for entire number
        x1 = min(left)
        x2 = max(left + width)
        y1 = min(top)
        y2 = max(top + height)
        bbox = [x1, y1, x2, y2]
        bboxes[i] = bbox
    
    return bboxes


# ==============================================================================
#                                                       ADD_DIGIT_BOUNDING_BOXES
# ==============================================================================
def add_digit_bounding_boxes(labels):
    """ Takes the raw labels dictionary which still contains the individual
        coordinate points for digit bounding boxes in the separate arrays:
            "top", "left", "width", "height"
        and modifies that dictionary to contain the following keys:
            bbox1, bbox2, bbox3, bbox4, bbox5
        Each with the full bounding box coordinates for the corresponding digit:
            [x1, y1, x2, y2]

    Args:
        labels: (dict)
    """
    # Limit to 5 digits
    for key in ["top", "left", "width", "height"]:
        labels[key] = labels[key][:, -5:]
    
    digit_bboxes = [None] * 5
    for i in range(5):
        digit_bboxes[i] = np.array([labels["left"][:, i],
                                    labels["top"][:, i],
                                    labels["left"][:, i] + labels["width"][:,
                                                           i],
                                    labels["top"][:, i] + labels["height"][:, i]
                                    ], dtype=np.int32).transpose()
    labels["bbox1"] = digit_bboxes[0]
    labels["bbox2"] = digit_bboxes[1]
    labels["bbox3"] = digit_bboxes[2]
    labels["bbox4"] = digit_bboxes[3]
    labels["bbox5"] = digit_bboxes[4]


# ==============================================================================
#                                                            CREATE_IMAGES_ARRAY
# ==============================================================================
def create_images_array(images_dir, labels, scale_factor=1.3, out_size=64):
    """Takes the dir containing the raw image files, along with the labels
    dictionary, and generates an array of images cropped close to the bounding
    box for the images.

    Args:
        labels:        The dictionary of arrays containing label data,
                       bounding box data and filenames of images.
        images_dir:    Where the images are located.
        scale_factor:  (float) Given a bounding box, how much should the size of
                       this box be scaled by before performing the crop?
        out_size:      (int) The length for the edges of the output square image

    Returns:
        Numpy array containing the images, and the updated labels
        The shape is: [num_sampes, out_size, out_size]
    """
    num_samples = labels["N"].shape[0]
    X = np.empty(shape=[num_samples, out_size, out_size], dtype=np.uint8)
    
    for i in range(num_samples):
        divs = num_samples / 25
        if (i % divs) == 0:
            print(".", end="")
            
        # LOAD UP RAW IMAGE
        f = os.path.join(images_dir, labels["file"][i])
        im = Image.open(f)
        
        # CONVERT TO GREYSCALE
        im = im.convert('L')
        
        # CROP TO REGION AROUND WHOLE BOUNDING BOX
        bbox = labels["bbox"][i]
        num_bboxes = 6  # 5 for each digit, and one whole bbox
        crop_coords = scale_bounding_box(bbox, scale=scale_factor,
                                         dtype=np.int32)
        # crop_offsets = np.concatenate([crop_coords[:2], crop_coords[:2]])
        crop_offsets = repeat_array(crop_coords[:2], n=2 * num_bboxes,
                                    axis=0)
        im = im.crop(crop_coords)
        # im.show()
        
        # RESIZE IMAGE
        resize_scale = np.array([out_size, out_size],
                                dtype=np.float32) / im.size
        resize_scale = repeat_array(resize_scale, n=2 * num_bboxes, axis=0)
        # resize_scale = np.concatenate([resize_scale, resize_scale])
        im = im.resize((out_size, out_size), Image.ANTIALIAS)
        
        # SHIFT AND RESCALE BOUNDING BOXES
        labels["bboxes"][i] = (labels["bboxes"][
                                   i] - crop_offsets) * resize_scale
        
        # CONVERT TO NUMPY ARRAY
        X[i] = np.asarray(im, dtype=np.uint8)
        
    print("", end="\n")
        
    return X, labels


# ==============================================================================
#                                                              GROW BOUNDING BOX
# ==============================================================================
def scale_bounding_box(bbox, scale, dtype=np.int32):
    """ Takes a bounding box and a scaling factor, and returns the coordinates
        of scaled bounding box.

    Args:
        bbox:   (array) [x1, y1, x2, y2]
        scale: (float)
    """
    bbox = bbox.astype(np.float32)
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    x_grow = ((bbox_width * scale) - bbox_width) / 2
    y_grow = ((bbox_height * scale) - bbox_height) / 2
    
    new_bbox = np.array([bbox[0] - x_grow,
                         bbox[1] - y_grow,
                         bbox[2] + x_grow,
                         bbox[3] + x_grow],
                        dtype=dtype
                        )
    
    return new_bbox


# ==============================================================================
#                                                          FIX_NULL_DIGIT_BBOXES
# ==============================================================================
def fix_null_digit_bboxes(labels, max_digits=5):
    """ makes the bounding box coordinates for null digits lie well outside the
        bounds of the image.

    Args:
        labels:     (dict) labels dictionary that must contain "N", and "bboxes"
        max_digits: (int) number of digits being represented by "bboxes"

    Returns:
        a copy of `labels` with the fixed bbox coordinates.
    """
    Y = copy.deepcopy(labels)
    for n in range(max_digits):
        # Index of bboxes column that this digit starts on
        i_bbox = (4 * (5 - n))
        # Find the rows of samples that have less than n number of digits
        samples = Y["N"] < n + 1
        # Replace the relevant items
        Y["bboxes"][samples, i_bbox:i_bbox + 4] = NULL_BBOX
    return Y
